"""Validate the EDT fallback against the semantics of SAM3's triton kernel.

sam3/model/edt.py seeds non-zero pixels with 1e18, runs a two-pass squared
Euclidean distance transform, then takes sqrt. That is exactly
cv2.distanceTransform(..., DIST_L2, DIST_MASK_PRECISE) over a 0/1 image, which
is what the upstream docstring claims. These checks pin that equivalence.
"""

import sys

sys.path.insert(0, r"F:\SAM3_OOOSplat")

import numpy as np
import torch

from windows_compat import _edt_fallback, _connected_components_fallback
import cv2

failures = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -> ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


edt = _edt_fallback(cv2, np, torch)
cc = _connected_components_fallback(cv2, np, torch)

print("== EDT: single zero pixel ==")
# Distance from every foreground pixel to the nearest zero pixel.
data = torch.ones(1, 5, 5, dtype=torch.float32)
data[0, 2, 2] = 0.0
got = edt(data)
want = torch.from_numpy(
    cv2.distanceTransform(data[0].numpy().astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
).float()[None]
check("matches cv2.distanceTransform", torch.allclose(got, want, atol=1e-5), f"max diff {(got - want).abs().max():.2e}")
check("zero pixel has distance 0", got[0, 2, 2].item() == 0.0, str(got[0, 2, 2].item()))
check("axis neighbour distance 1", got[0, 2, 3].item() == 1.0, str(got[0, 2, 3].item()))
check("diagonal neighbour distance sqrt(2)", abs(got[0, 1, 1].item() - 2**0.5) < 1e-5, str(got[0, 1, 1].item()))
check("far corner distance sqrt(8)", abs(got[0, 0, 0].item() - (8**0.5)) < 1e-5, str(got[0, 0, 0].item()))

print("\n== EDT: two zero pixels take the nearer one ==")
data = torch.ones(1, 7, 7, dtype=torch.float32)
data[0, 3, 1] = 0.0
data[0, 3, 5] = 0.0
got = edt(data)
check("midpoint between the two zeros is 2", abs(got[0, 3, 3].item() - 2.0) < 1e-5, str(got[0, 3, 3].item()))
# Corner (0,0) to the nearest zero (3,1) is sqrt(3^2+1^2)=sqrt(10).
check("corner distance sqrt(10)", abs(got[0, 0, 0].item() - (10**0.5)) < 1e-3, str(got[0, 0, 0].item()))
check("every pixel matches cv2 exactly", torch.allclose(got, torch.from_numpy(
    cv2.distanceTransform(data[0].numpy().astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
).float()[None], atol=1e-5))

print("\n== EDT: shape / dtype / device / batching ==")
data = torch.zeros(3, 12, 20, dtype=torch.float32)
data[:, 4:9, 6:15] = 1.0
got = edt(data)
check("preserves shape", tuple(got.shape) == (3, 12, 20), str(tuple(got.shape)))
check("preserves dtype", got.dtype == data.dtype, str(got.dtype))
check("preserves device", got.device == data.device, str(got.device))
check("3D shape is not silently reshaped", tuple(got.shape) == (3, 12, 20))

print("\n== EDT: boolean and uint8 masks (SAM3 passes both) ==")
mask_bool = torch.zeros(1, 9, 9, dtype=torch.bool)
mask_bool[0, 2:7, 2:7] = True
got_bool = edt(mask_bool)
want_bool = edt(mask_bool.to(torch.float32))
check("bool mask == float mask", torch.allclose(got_bool.float(), want_bool, atol=1e-5))
mask_u8 = torch.zeros(1, 9, 9, dtype=torch.uint8)
mask_u8[0, 2:7, 2:7] = 1
check("uint8 mask == float mask", torch.allclose(edt(mask_u8).float(), want_bool, atol=1e-5))
check("bool input returns float distances", got_bool.dtype == torch.float32, str(got_bool.dtype))

print("\n== EDT: a foreground ring around a zero block ==")
# Foreground field with a zero block punched out: the ring immediately around
# the block is exactly 1 away, growing with distance further out.
data = torch.ones(1, 32, 32, dtype=torch.float32)
data[0, 4:28, 4:28] = 0.0
got = edt(data)
check("zero block itself is 0", got[0, 16, 16].item() == 0.0, str(got[0, 16, 16].item()))
check("pixel just outside the block is 1", abs(got[0, 3, 16].item() - 1.0) < 1e-5, str(got[0, 3, 16].item()))
check("two steps out is 2", abs(got[0, 2, 16].item() - 2.0) < 1e-5, str(got[0, 2, 16].item()))
check("corner matches cv2", torch.allclose(got, torch.from_numpy(
    cv2.distanceTransform(data[0].numpy().astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
).float()[None], atol=1e-5))

print("\n== EDT: invalid input is rejected ==")
try:
    edt(torch.ones(4, 4, dtype=torch.float32))
    check("2D input raises ValueError", False, "no exception")
except ValueError as exc:
    check("2D input raises ValueError", "3D" in str(exc), str(exc)[:70])

print("\n== EDT: non-contiguous input (SAM3 slices tensors) ==")
big = torch.zeros(1, 20, 20, dtype=torch.float32)
big[0, 5:15, 5:15] = 1.0
view = big[:, ::2, ::2]
check("non-contiguous view handled", tuple(edt(view).shape) == tuple(view.shape), f"{tuple(view.shape)} contiguous={view.is_contiguous()}")

print("\n== connected components ==")
t = torch.zeros(1, 10, 10)
t[0, 1:4, 1:4] = 1
t[0, 6:9, 6:9] = 1
labels, counts = cc(t)
check("finds 2 blobs + background", int(labels.max()) == 2, f"labels max={int(labels.max())}")
check("blob pixel counts", sorted(counts[0].unique().tolist()) == [0, 9], str(sorted(counts[0].unique().tolist())))
check("returns on the input device", labels.device == t.device and counts.device == t.device)

t4 = torch.zeros(2, 1, 8, 8)
t4[0, 0, 1:3, 1:3] = 1
t4[1, 0, 5:7, 5:7] = 1
lab4, cnt4 = cc(t4)
check("4D input keeps the channel dim", tuple(lab4.shape) == (2, 1, 8, 8), str(tuple(lab4.shape)))

print()
if failures:
    print(f"FAILED {len(failures)}: {failures}")
    raise SystemExit(1)
print("WINDOWS_COMPAT_OK  (all checks passed)")