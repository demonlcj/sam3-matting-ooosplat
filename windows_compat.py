"""Windows compatibility helpers for Meta's SAM3.

SAM3 is Linux/CUDA-first. On Windows two separate things break, and they break
at different moments, so they need separate fixes:

1. **Import time.** ``sam3/__init__.py`` -> ``sam3_tracker_utils.py`` ->
   ``sam3/model/edt.py`` does a module-scope ``import triton``. Triton ships no
   Windows wheels (only Linux/macOS), so ``import sam3`` raises
   ``ModuleNotFoundError: No module named 'triton'`` before any of our code
   runs. :func:`install_triton_import_stub` registers a placeholder so the
   import graph can be built.

2. **Call time.** ``sam3_tracker_utils.py`` calls ``edt_triton`` unconditionally
   and also routes mask post-processing through
   ``sam3.perflib.connected_components``. A placeholder that only satisfies the
   import is not enough -- these have to be replaced with real implementations.
   :func:`install_sam3_cpu_fallbacks` does that.

Both helpers are no-ops when Triton is genuinely importable, so the same code
path is safe to call unconditionally.

The EDT replacement is faithful rather than approximate: the kernel in
``sam3/model/edt.py`` seeds foreground pixels with ``1e18``, runs a two-pass
squared Euclidean distance transform, then takes ``sqrt`` -- which is exactly
``cv2.distanceTransform(..., DIST_L2, DIST_MASK_PRECISE)`` over a 0/1 image, as
the upstream docstring itself notes.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types

__all__ = [
    "install_triton_import_stub",
    "install_sam3_cpu_fallbacks",
    "triton_available",
]


def triton_available() -> bool:
    """Return True when a real Triton installation can be imported."""
    if "triton" in sys.modules:
        return not getattr(sys.modules["triton"], "_sam3_windows_stub", False)
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return True


def install_triton_import_stub() -> bool:
    """Allow ``import triton`` to succeed so SAM3's import graph can be built.

    Triton has no Windows wheels, so ``sam3.model.edt`` cannot be imported at
    all. SAM3 only uses Triton to *define* ``@triton.jit`` kernels at module
    scope, so satisfying the decorator is enough to get through the import.
    Nothing here is meant to execute: every callable raises, so if a real
    kernel is ever reached without a CPU fallback in place the failure is loud
    rather than a silent wrong answer.

    Returns True if a stub was installed, False if real Triton was available.
    """
    if triton_available():
        return False

    if "triton" in sys.modules:
        return True

    message = (
        "Triton is not available on this platform (it publishes no Windows "
        "wheels). This code path should have been replaced by a CPU fallback; "
        "see windows_compat.install_sam3_cpu_fallbacks."
    )

    class _JitStub:
        """Stands in for ``triton.jit``; usable bare or parameterized."""

        def __call__(self, fn=None, **kwargs):
            if fn is None:
                return self

            def _unavailable(*args, __fn=fn, **kw):
                raise RuntimeError(
                    f"{getattr(__fn, '__name__', 'triton_kernel')} is a Triton "
                    f"kernel and cannot run here. {message}"
                )

            _unavailable.__name__ = getattr(fn, "__name__", "triton_kernel")
            _unavailable.__doc__ = getattr(fn, "__doc__", None)
            _unavailable.__wrapped__ = fn
            return _unavailable

    def _unavailable(*args, **kwargs):
        raise RuntimeError(message)

    triton = types.ModuleType("triton")
    triton.__version__ = "0.0.0+sam3-windows-stub"
    triton._sam3_windows_stub = True
    triton.jit = _JitStub()
    triton.autotune = _JitStub()
    triton.heuristics = _JitStub()
    triton.Config = _unavailable
    triton.cdiv = _unavailable
    triton.next_power_of_2 = _unavailable

    # ``sam3/model/edt.py`` does ``import triton.language as tl``, so the
    # submodule must exist in sys.modules for that binding to resolve.
    language = types.ModuleType("triton.language")
    language.constexpr = type(
        "constexpr", (), {"__init__": lambda self, value=None: setattr(self, "value", value)}
    )
    for name in ("program_id", "load", "store", "num_programs", "arange", "exp", "log", "sqrt", "abs"):
        setattr(language, name, _unavailable)

    triton.language = language

    # torch.cuda probes for Triton with ``importlib.util.find_spec("triton")``
    # during its own import. A hand-built ModuleType has ``__spec__ = None``,
    # which makes find_spec raise ValueError and breaks CUDA initialisation
    # with "triton.__spec__ is None". Give both modules a real ModuleSpec so
    # find_spec returns cleanly and torch concludes Triton is absent.
    for name, module in (("triton", triton), ("triton.language", language)):
        spec = importlib.machinery.ModuleSpec(name, None)
        spec.has_location = False
        module.__spec__ = spec
        module.__loader__ = None

    sys.modules["triton"] = triton
    sys.modules["triton.language"] = language
    return True


def _connected_components_fallback(cv2, np, torch):
    def connected_components(input_tensor):
        original_shape = tuple(input_tensor.shape)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(1)
        if input_tensor.dim() != 4 or input_tensor.shape[1] != 1:
            raise ValueError("Input tensor must be (B, H, W) or (B, 1, H, W).")
        binary = input_tensor.detach().to("cpu").numpy()[:, 0] != 0
        labels_batch = []
        counts_batch = []
        for image in binary:
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                image.astype(np.uint8), connectivity=8
            )
            labels = labels.astype(np.int64, copy=False)
            counts = np.zeros_like(labels, dtype=np.int64)
            if count > 0:
                areas = stats[:, cv2.CC_STAT_AREA].astype(np.int64, copy=False)
                areas[0] = 0
                counts = areas[labels]
            labels_batch.append(labels)
            counts_batch.append(counts)
        device = input_tensor.device
        labels_tensor = torch.from_numpy(np.stack(labels_batch, axis=0)).to(device)
        counts_tensor = torch.from_numpy(np.stack(counts_batch, axis=0)).to(device)
        if len(original_shape) == 3:
            return labels_tensor, counts_tensor
        return labels_tensor.unsqueeze(1), counts_tensor.unsqueeze(1)

    return connected_components


def _edt_fallback(cv2, np, torch):
    def edt_triton(data: "torch.Tensor") -> "torch.Tensor":
        """Batched Euclidean distance transform via OpenCV.

        Matches ``sam3.model.edt.edt_triton``: the distance from every non-zero
        pixel to the nearest zero pixel.

        The upstream kernel always produces ``torch.float`` (its scratch
        buffers are allocated as ``dtype=torch.float`` and the result comes
        from ``torch.where(data, 1e18, 0.0).sqrt()``), so the dtype is fixed to
        float32 here rather than inherited from the input. Inheriting it would
        return a bool tensor for a bool mask, and the caller does float
        comparisons (``fn_max > fp_max``) that would silently collapse.
        """
        if data.dim() != 3:
            raise ValueError(f"Expected a 3D (B, H, W) tensor, got {tuple(data.shape)}.")
        batch, height, width = data.shape
        binary = (data.detach() != 0).to("cpu").numpy()
        result = np.empty((batch, height, width), dtype=np.float32)
        for index in range(batch):
            result[index] = cv2.distanceTransform(
                binary[index].astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            )
        return torch.from_numpy(result).to(device=data.device, dtype=torch.float32)

    return edt_triton


def install_sam3_cpu_fallbacks() -> bool:
    """Replace SAM3's Triton kernels with OpenCV CPU implementations.

    Must run after :func:`install_triton_import_stub` and after ``sam3`` is
    importable, because the patched functions are looked up on the SAM3 modules
    that call them.

    Returns True if anything was patched.
    """
    import cv2
    import numpy as np
    import torch

    patched = False

    try:
        import sam3.perflib.connected_components as components
    except ImportError:
        components = None
    if components is not None:
        components.connected_components = _connected_components_fallback(cv2, np, torch)
        patched = True

    # ``sam3_tracker_utils`` does ``from sam3.model.edt import edt_triton``, so
    # the name is bound in *its* namespace -- patching sam3.model.edt alone
    # would not affect the call sites.
    try:
        import sam3.model.sam3_tracker_utils as tracker_utils
    except ImportError:
        tracker_utils = None
    if tracker_utils is not None and hasattr(tracker_utils, "edt_triton"):
        tracker_utils.edt_triton = _edt_fallback(cv2, np, torch)
        patched = True

    return patched