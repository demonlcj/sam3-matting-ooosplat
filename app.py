from __future__ import annotations

import json
import os
import shutil
import sys
import subprocess
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRect, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSlider, QSpinBox, QStyle, QTabWidget, QToolButton, QVBoxLayout,
    QWidget,
)

APP_DIR = Path(__file__).resolve().parent
SAM3_ROOT = Path(os.environ.get("SAM3_ROOT", r"F:\SAM3"))
SAM3_CHECKPOINT = Path(os.environ.get("SAM3_CHECKPOINT", SAM3_ROOT / "sam3.pt"))
MATTING_ROOT = Path(os.environ.get("MATTING_ROOT", str(APP_DIR / "Matting-Anything")))
MATTING_CHECKPOINT = Path(os.environ.get("MATTING_CHECKPOINT", MATTING_ROOT / "checkpoints" / "mam_sam_vitb.pth"))
ACTIVE_OUTPUT_DIR = None
COLORS = {
    "bg": "#11151B", "panel": "#1B222C", "panel_2": "#202A35", "line": "#34414E",
    "text": "#E4EBEF", "muted": "#8B9AA7", "teal": "#31D0B0", "teal_dim": "#173D3C",
    "amber": "#E9B85A", "red": "#E86B73",
}


def preferred_font(candidates, fallback):
    installed = set(QFontDatabase.families())
    return next((name for name in candidates if name in installed), fallback)


def _write_crash_log(exc_type, exc_value, exc_traceback):
    """Keep Python exceptions visible when the app is launched with pythonw."""
    import traceback as _traceback
    try:
        log_root = ACTIVE_OUTPUT_DIR if ACTIVE_OUTPUT_DIR is not None else APP_DIR
        log_root.mkdir(parents=True, exist_ok=True)
        log_path = log_root / "sam3_ooosplat_error.log"
        with log_path.open("a", encoding="utf-8") as handle:
            _traceback.print_exception(exc_type, exc_value, exc_traceback, file=handle)
    except OSError:
        pass


sys.excepthook = _write_crash_log


def cv_to_qimage(frame_bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format.Format_RGB888).copy()


def frame_to_qimage(frame: np.ndarray, mode: str, mask: np.ndarray | None) -> QImage:
    if mode == "mask":
        image = np.zeros((*frame.shape[:2], 3), dtype=np.uint8)
        if mask is not None:
            image[mask.astype(bool)] = (49, 208, 176)
        return cv_to_qimage(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if mode == "alpha":
        if mask is None:
            alpha = np.zeros(frame.shape[:2], dtype=np.uint8)
        elif np.issubdtype(mask.dtype, np.floating):
            alpha = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
        elif mask.dtype == np.bool_:
            alpha = mask.astype(np.uint8) * 255
        else:
            alpha = np.clip(mask, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        checker = np.full(rgb.shape, 36, dtype=np.uint8)
        checker[::2, ::2] = 54; checker[1::2, 1::2] = 54
        opacity = alpha.astype(np.float32)[..., None] / 255.0; composite = (rgb.astype(np.float32) * opacity + checker.astype(np.float32) * (1.0 - opacity)).astype(np.uint8)
        return QImage(composite.data, composite.shape[1], composite.shape[0], composite.shape[1] * 3, QImage.Format.Format_RGB888).copy()
    if mode == "overlay" and mask is not None:
        image = frame.copy(); tint = np.zeros_like(image); tint[:, :, 1] = 190; active = mask.astype(bool)
        image[active] = (image[active] * 0.48 + tint[active] * 0.52).astype(np.uint8)
        return cv_to_qimage(image)
    return cv_to_qimage(frame)


def _warn_if_not_h264(video_path):
    """SAM3's video loader only decodes H.264 MP4; anything else fails later
    with "Only MP4 video and JPEG folder are supported", which does not tell the
    user their codec is the problem. OpenCV's default writer produces mp4v, so
    this is a common trap. Warns early instead of failing after model load.
    """
    probe = shutil.which("ffprobe")
    if not probe: return
    try:
        result = subprocess.run(
            [probe, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=codec_name", "-of", "default=nw=1:nk=1", str(video_path)],
            capture_output=True, text=True, timeout=15,
        )
        codec = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return
    if codec and codec.lower() not in ("h264", "avc1"):
        raise RuntimeError(
            f"SAM3 只支持 H.264 编码的 MP4，当前文件是 {codec}。\n"
            "请先转码：\n"
            f'  ffmpeg -i "{video_path}" -c:v libx264 -pix_fmt yuv420p '
            "-crf 20 -preset veryfast out.mp4"
        )


def _install_windows_sam3_fallbacks():
    """Replace SAM3's Triton-only kernels with OpenCV CPU implementations.

    Triton publishes no Windows wheels, so SAM3 cannot even be imported without
    help. ``windows_compat`` handles both the import-time stub and the
    call-time replacements; see that module for the full rationale.
    """
    try:
        from windows_compat import install_sam3_cpu_fallbacks
    except ImportError:
        return
    install_sam3_cpu_fallbacks()


class PreviewCanvas(QFrame):
    point_added = Signal(float, float, bool)

    def __init__(self):
        super().__init__(); self.setMinimumSize(560, 380); self.setObjectName("previewCanvas")
        self._frame = None; self._mask = None; self._mode = "original"; self._points = []; self._display_rect = QRect(); self._preview_image = None

    def set_frame(self, frame, mask=None):
        self._frame, self._mask = frame, mask; self._refresh_image()

    def set_mask(self, mask):
        self._mask = mask; self._refresh_image()

    def set_mode(self, mode):
        self._mode = mode; self._refresh_image()

    def set_points(self, points):
        self._points = list(points); self.update()

    def _refresh_image(self):
        self._preview_image = None if self._frame is None else frame_to_qimage(self._frame, self._mode, self._mask); self.update()

    def paintEvent(self, event):
        painter = QPainter(self); painter.fillRect(self.rect(), QColor(COLORS["bg"]))
        if self._preview_image is None:
            painter.setPen(QColor(COLORS["muted"])); painter.setFont(QFont("Microsoft YaHei UI", 12)); painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "选择视频以载入预览"); painter.end(); return
        target_size = QSize(max(1, self.width() - 32), max(1, self.height() - 32))
        pixmap = QPixmap.fromImage(self._preview_image).scaled(target_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        x, y = (self.width() - pixmap.width()) // 2, (self.height() - pixmap.height()) // 2; self._display_rect = QRect(x, y, pixmap.width(), pixmap.height()); painter.drawPixmap(self._display_rect, pixmap)
        if self._mode == "original":
            for px, py, positive in self._points:
                painter.setPen(QPen(QColor(COLORS["teal"] if positive else COLORS["red"]), 3)); painter.drawEllipse(x + int(px * pixmap.width()) - 7, y + int(py * pixmap.height()) - 7, 14, 14)
        painter.end()

    def resizeEvent(self, event):
        self.update(); super().resizeEvent(event)

    def mousePressEvent(self, event):
        if self._frame is None or not self._display_rect.contains(event.position().toPoint()): return
        x = (event.position().x() - self._display_rect.x()) / max(1, self._display_rect.width()); y = (event.position().y() - self._display_rect.y()) / max(1, self._display_rect.height())
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton): self.point_added.emit(float(x), float(y), event.button() != Qt.MouseButton.RightButton)


class TimelineStrip(QFrame):
    frame_selected = Signal(int)

    def __init__(self):
        super().__init__(); self.setFixedHeight(74); self.setObjectName("timelineStrip"); self._thumbs = []; self._frame_count = 0; self._current = 0; self._mask_frames = set(); self._complete = False

    def set_video(self, path, frame_count):
        self._thumbs.clear(); self._frame_count = frame_count; self._current = 0; self._complete = False; self._mask_frames.clear()
        cap = cv2.VideoCapture(path); sample_count = min(12, max(1, frame_count)); indexes = np.linspace(0, max(0, frame_count - 1), sample_count, dtype=int)
        for index in indexes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index)); ok, frame = cap.read()
            if ok: self._thumbs.append((int(index), cv_to_qimage(frame)))
        cap.release(); self.update()

    def set_frames(self, frame_paths):
        paths = list(frame_paths); self._thumbs.clear(); self._frame_count = len(paths); self._current = 0; self._complete = False; self._mask_frames.clear()
        if paths:
            for index in np.linspace(0, len(paths) - 1, min(12, len(paths)), dtype=int):
                frame = cv2.imread(str(paths[int(index)]), cv2.IMREAD_COLOR)
                if frame is not None: self._thumbs.append((int(index), cv_to_qimage(frame)))
        self.update()

    def set_current(self, index):
        self._current = index; self.update()

    def set_mask_frames(self, frame_indexes, complete=True):
        self._mask_frames = set(frame_indexes); self._complete = complete; self.update()

    def paintEvent(self, event):
        painter = QPainter(self); painter.fillRect(self.rect(), QColor("#131A22"))
        if not self._thumbs:
            painter.setPen(QColor(COLORS["muted"])); painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "FRAME THUMBNAILS"); painter.end(); return
        gap, margin = 5, 7; width = max(30, (self.width() - margin * 2 - gap * (len(self._thumbs) - 1)) // len(self._thumbs)); height = self.height() - 14
        for position, (index, image) in enumerate(self._thumbs):
            rect = QRect(margin + position * (width + gap), 7, width, height); pixmap = QPixmap.fromImage(image).scaled(rect.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation); painter.drawPixmap(rect, pixmap, pixmap.rect())
            selected = abs(index - self._current) <= max(1, self._frame_count // max(2, len(self._thumbs) * 2)); color = COLORS["teal"] if selected else COLORS["line"]; painter.setPen(QPen(QColor(color), 2 if selected else 1)); painter.drawRect(rect.adjusted(0, 0, -1, -1))
            if index == 0: painter.fillRect(QRect(rect.x(), rect.bottom() - 3, rect.width(), 3), QColor(COLORS["teal"]))
            if self._complete and index not in self._mask_frames: painter.fillRect(QRect(rect.right() - 5, rect.y(), 5, 5), QColor(COLORS["red"]))
        painter.end()

    def mousePressEvent(self, event):
        if not self._thumbs: return
        margin = 7
        ratio = min(1.0, max(0.0, (event.position().x() - margin) / max(1, self.width() - margin * 2)))
        self.frame_selected.emit(int(round(ratio * max(0, self._frame_count - 1))))


class SeekSlider(QSlider):
    """Video seek slider with direct click-to-position and smooth dragging."""

    def _value_from_x(self, x):
        handle = self.style().pixelMetric(QStyle.PixelMetric.PM_SliderLength, None, self)
        span = max(1, self.width() - handle)
        position = int(round(min(span, max(0.0, x - handle / 2))))
        return QStyle.sliderValueFromPosition(
            self.minimum(), self.maximum(), position, span, self.invertedAppearance()
        )

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        self.setSliderDown(True)
        self.setValue(self._value_from_x(event.position().x()))
        event.accept()

    def mouseMoveEvent(self, event):
        if self.isSliderDown() and event.buttons() & Qt.MouseButton.LeftButton:
            self.setValue(self._value_from_x(event.position().x()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.isSliderDown():
            self.setValue(self._value_from_x(event.position().x()))
            self.setSliderDown(False)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class TrackWorker(QObject):
    progress = Signal(int, str); finished = Signal(str); failed = Signal(str)

    def __init__(self, video_path, points, output_dir):
        super().__init__(); self.video_path, self.points, self.output_dir = video_path, points, output_dir

    def run(self):
        try: self._run_impl()
        except Exception: self.failed.emit(traceback.format_exc())

    def _run_impl(self):
        # pythonw.exe has no console streams. SAM3's video loader uses tqdm,
        # which expects stderr.write even though its progress is not needed here.
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w", encoding="utf-8")
        if sys.stdout is None:
            sys.stdout = open(os.devnull, "w", encoding="utf-8")
        if not SAM3_ROOT.exists(): raise FileNotFoundError(f"SAM3_ROOT 不存在: {SAM3_ROOT}")
        if not SAM3_CHECKPOINT.exists(): raise FileNotFoundError(f"SAM3 checkpoint 不存在: {SAM3_CHECKPOINT}")
        sys.path.insert(0, str(SAM3_ROOT))
        # SAM3 imports triton at module scope and triton has no Windows wheels,
        # so the stub must be registered before the first sam3 import.
        try:
            from windows_compat import install_triton_import_stub
        except ImportError:
            pass
        else:
            install_triton_import_stub()
        try:
            from sam3.model_builder import build_sam3_video_model
        except ImportError as exc:
            raise RuntimeError(
                "SAM3 视频依赖未完整安装。请在 SAM3 环境中运行：\n"
                f"{sys.executable} -m pip install psutil\n\n原始错误：{exc}"
            ) from exc
        _install_windows_sam3_fallbacks()
        self.output_dir.mkdir(parents=True, exist_ok=True); dirs = [self.output_dir / name for name in ("mask_frames", "rgba_frames", "overlay_frames")]
        for path in dirs: path.mkdir(exist_ok=True)
        # SAM3's load_video_frames does ``isinstance(video_path, str)`` and then
        # checks the extension, so a Path silently fails the .mp4 test and raises
        # a misleading NotImplementedError. Normalise before handing it over.
        if not isinstance(self.video_path, str): self.video_path = str(self.video_path)
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened(): raise RuntimeError(f"无法打开视频: {self.video_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 24.0; width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
        if Path(self.video_path).suffix.lower() == ".mp4": _warn_if_not_h264(self.video_path)
        self.progress.emit(1, "加载 SAM3 模型…")
        # Use SAM3's official SAM2-compatible interactive tracker API. The
        # higher-level video predictor is for text/detection prompts and does
        # not register point-only prompts in its propagation state.
        import torch
        sam3_model = build_sam3_video_model(checkpoint_path=str(SAM3_CHECKPOINT), load_from_HF=False, compile=False)
        predictor = sam3_model.tracker
        predictor.backbone = sam3_model.detector.backbone
        inference_state = predictor.init_state(video_path=self.video_path, offload_video_to_cpu=False, offload_state_to_cpu=False)
        points = torch.tensor([[x, y] for x, y, _ in self.points], dtype=torch.float32)
        labels = torch.tensor([1 if p else 0 for _, _, p in self.points], dtype=torch.int32)
        _, _, _, initial_masks = predictor.add_new_points(inference_state=inference_state, frame_idx=0, obj_id=1, points=points, labels=labels, clear_old_points=True, rel_coordinates=True)
        outputs = {0: self._extract_tracker_mask(initial_masks)}
        for index, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(inference_state, start_frame_idx=0, max_frame_num_to_track=frame_count, reverse=False, tqdm_disable=True, propagate_preflight=True):
            outputs[int(index)] = self._extract_tracker_mask(video_res_masks)
            self.progress.emit(min(88, 5 + int(83 * (index + 1) / max(frame_count, 1))), f"跟踪第 {index + 1}/{frame_count} 帧")
        self.progress.emit(90, "导出预览、透明帧和 RGBA 视频…"); self._export(outputs, fps, width, height)
        # Let the worker unwind naturally. Explicitly deleting the tracker and
        # flushing CUDA here can race with PyTorch's background cleanup on Windows.
        self.progress.emit(100, "SAM3 跟踪完成"); self.finished.emit(str(self.output_dir))

    @staticmethod
    def _extract_mask(outputs):
        masks = outputs.get("out_binary_masks")
        if masks is None: return None
        masks = np.asarray(masks); masks = masks[:, 0] if masks.ndim == 4 else masks
        return masks[0].astype(bool) if masks.shape[0] else None

    @staticmethod
    def _extract_tracker_mask(video_res_masks):
        if video_res_masks is None or len(video_res_masks) == 0:
            return None
        masks = video_res_masks.detach().cpu().numpy() if hasattr(video_res_masks, "detach") else np.asarray(video_res_masks)
        while masks.ndim > 2:
            masks = masks[0]
        return masks > 0.0

    @staticmethod
    def _encode_rgba_video(rgba_dir, output_dir, fps):
        """Encode the RGBA PNG sequence using a codec that preserves alpha."""
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return None, "FFmpeg 未找到，已保留 RGBA PNG 序列"
        input_pattern = str(rgba_dir / "%06d.png")
        attempts = [
            (output_dir / "rgba_video.webm", [
                "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "18",
            ]),
            (output_dir / "rgba_video.mov", [
                "-c:v", "qtrle", "-pix_fmt", "argb",
            ]),
        ]
        for output_path, codec_args in attempts:
            command = [
                ffmpeg, "-y", "-loglevel", "error", "-framerate", f"{max(float(fps), 1.0):.6f}",
                "-i", input_pattern, "-vf", "format=rgba", *codec_args,
                "-metadata:s:v:0", "alpha_mode=1", str(output_path),
            ]
            try:
                result = subprocess.run(command, capture_output=True, check=False)
            except OSError as exc:
                return None, f"FFmpeg 启动失败，已保留 RGBA PNG 序列：{exc}"
            if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
                return output_path, f"已导出透明视频：{output_path.name}"
            output_path.unlink(missing_ok=True)
        return None, "FFmpeg 没有可用的 Alpha 编码器，已保留 RGBA PNG 序列"

    def _export(self, masks, fps, width, height):
        mask_dir, rgba_dir, overlay_dir = [self.output_dir / name for name in ("mask_frames", "rgba_frames", "overlay_frames")]; cap = cv2.VideoCapture(self.video_path); writer = cv2.VideoWriter(str(self.output_dir / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)); index = 0; total = max(1, len(masks))
        while True:
            ok, frame = cap.read()
            if not ok: break
            mask = masks.get(index); mask = np.zeros((height, width), dtype=bool) if mask is None else mask
            if mask.shape != (height, width): mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            mask_u8 = mask.astype(np.uint8) * 255; cv2.imwrite(str(mask_dir / f"{index:06d}.png"), mask_u8); rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA); rgba[:, :, 3] = mask_u8; rgba[~mask, :3] = 0; cv2.imwrite(str(rgba_dir / f"{index:06d}.png"), rgba)
            overlay = frame.copy(); tint = np.zeros_like(frame); tint[:, :, 1] = 190; overlay[mask] = (overlay[mask] * 0.48 + tint[mask] * 0.52).astype(np.uint8); writer.write(overlay); cv2.imwrite(str(overlay_dir / f"{index:06d}.jpg"), overlay); index += 1; self.progress.emit(min(97, 90 + int(7 * index / total)), f"导出第 {index}/{total} 帧")
        cap.release(); writer.release()
        self.progress.emit(98, "编码 RGBA Alpha 视频…")
        rgba_video, rgba_message = self._encode_rgba_video(rgba_dir, self.output_dir, fps)
        metadata = {"source_video": str(Path(self.video_path).resolve()), "frame_count": index, "fps": fps, "width": width, "height": height, "points": [{"x": x, "y": y, "positive": bool(p)} for x, y, p in self.points], "rgba_video": str(rgba_video.name) if rgba_video else None, "rgba_video_status": rgba_message, "rgba_frames": "rgba_frames"}
        (self.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


class MattingWorker(QObject):
    progress = Signal(int, str); finished = Signal(str); failed = Signal(str)

    def __init__(self, source_video, sam3_output_dir, output_dir, checkpoint_path, edge_width=4):
        super().__init__(); self.source_video = Path(source_video); self.sam3_output_dir = Path(sam3_output_dir); self.output_dir = Path(output_dir); self.checkpoint_path = Path(checkpoint_path); self.edge_width = max(1, int(edge_width))

    def run(self):
        try: self._run_impl()
        except Exception: self.failed.emit(traceback.format_exc())

    def _run_impl(self):
        if not MATTING_ROOT.exists(): raise FileNotFoundError(f"Matting Anything 项目不存在: {MATTING_ROOT}")
        if not self.checkpoint_path.exists() or self.checkpoint_path.stat().st_size < 100 * 1024 * 1024:
            raise FileNotFoundError(f"MAM checkpoint 不存在或未下载完整: {self.checkpoint_path}")
        sys.path.insert(0, str(MATTING_ROOT)); sys.path.insert(0, str(MATTING_ROOT / "segment-anything"))
        import torch
        import torch.nn.functional as F
        from segment_anything.utils.transforms import ResizeLongestSide
        import networks
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dir.mkdir(parents=True, exist_ok=True); alpha_dir = self.output_dir / "matting_frames"; rgba_dir = self.output_dir / "matting_rgba_frames"; alpha_dir.mkdir(exist_ok=True); rgba_dir.mkdir(exist_ok=True)
        self.progress.emit(2, "加载 Matting Anything 模型…")
        model = networks.get_generator_m2m(seg="sam", m2m="sam_decoder_deep").to(device)
        checkpoint = torch.load(str(self.checkpoint_path), map_location=device)
        state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if isinstance(state, dict) and state:
            first_key = next(iter(state))
            if first_key.startswith("module."): state = {key[7:]: value for key, value in state.items()}
        model.load_state_dict(state, strict=True); model.eval()
        transform = ResizeLongestSide(1024)
        cap = cv2.VideoCapture(str(self.source_video)); fps = cap.get(cv2.CAP_PROP_FPS) or 24.0; frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not cap.isOpened(): raise RuntimeError(f"无法打开视频: {self.source_video}")
        writer = cv2.VideoWriter(str(self.output_dir / "matting_preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)); index = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            mask_path = self.sam3_output_dir / "mask_frames" / f"{index:06d}.png"; mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
            if mask is None: raise FileNotFoundError(f"缺少 SAM3 Mask: {mask_path.name}")
            if mask.shape != (height, width): mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            alpha = self._infer_frame(model, transform, frame, mask, device, torch, F)
            alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8); cv2.imwrite(str(alpha_dir / f"{index:06d}.png"), alpha_u8)
            rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA); rgba[:, :, 3] = alpha_u8; rgba[alpha_u8 == 0, :3] = 0; cv2.imwrite(str(rgba_dir / f"{index:06d}.png"), rgba)
            preview = frame.copy(); tint = np.zeros_like(frame); tint[:, :, 1] = 190; active = alpha_u8 > 8; preview[active] = (preview[active] * 0.48 + tint[active] * 0.52).astype(np.uint8); writer.write(preview)
            index += 1; self.progress.emit(min(88, 5 + int(82 * index / max(frame_count, 1))), f"Matting 第 {index}/{frame_count} 帧")
        cap.release(); writer.release(); self.progress.emit(90, "编码 Matting 预览与 RGBA 视频…"); rgba_video, rgba_message = TrackWorker._encode_rgba_video(rgba_dir, self.output_dir, fps); self.progress.emit(98, "写入 Matting 成果清单…")
        metadata = {"source_video": str(self.source_video.resolve()), "frame_count": index, "fps": fps, "width": width, "height": height, "model": "Matting Anything MAM ViT-B", "checkpoint": str(self.checkpoint_path.resolve()), "edge_width": self.edge_width, "rgba_video": rgba_video.name if rgba_video else None, "rgba_video_status": rgba_message, "alpha_frames": "matting_frames", "rgba_frames": "matting_rgba_frames"}
        (self.output_dir / "matting_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"); self.progress.emit(100, "Matting Alpha 细化完成"); self.finished.emit(str(self.output_dir))

    def _infer_frame(self, model, transform, frame, mask, device, torch, F):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); original_size = rgb.shape[:2]; ys, xs = np.where(mask > 127)
        if len(xs) == 0: return np.zeros(original_size, dtype=np.float32)
        pad = self.edge_width * 3; x1, x2 = max(0, int(xs.min()) - pad), min(original_size[1] - 1, int(xs.max()) + pad); y1, y2 = max(0, int(ys.min()) - pad), min(original_size[0] - 1, int(ys.max()) + pad)
        image = transform.apply_image(rgb); image = torch.as_tensor(image, device=device).permute(2, 0, 1).contiguous().float(); image = (image - torch.tensor([123.675, 116.28, 103.53], device=device).view(3, 1, 1)) / torch.tensor([58.395, 57.12, 57.375], device=device).view(3, 1, 1); pad_size = image.shape[-2:]; image = F.pad(image, (0, 1024 - image.shape[-1], 0, 1024 - image.shape[-2]))
        bbox = transform.apply_boxes(np.array([[x1, y1, x2, y2]], dtype=np.float32), original_size); bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device).unsqueeze(0); sample = {"image": image.unsqueeze(0), "bbox": bbox, "ori_shape": original_size, "pad_shape": pad_size}
        with torch.no_grad():
            _, pred, _ = model.forward_inference(sample); alpha = pred["alpha_os8"][..., :pad_size[0], :pad_size[1]]; alpha = F.interpolate(alpha, original_size, mode="bilinear", align_corners=False)[0, 0].clamp(0, 1).cpu().numpy()
        binary = mask > 127; kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.edge_width * 2 + 1, self.edge_width * 2 + 1)); core = cv2.erode(binary.astype(np.uint8), kernel) > 0; outside = cv2.erode((~binary).astype(np.uint8), kernel) > 0; alpha[outside] = 0; alpha[core] = 1; return alpha


class OoosplatExportWorker(QObject):
    progress = Signal(int, str); finished = Signal(str); failed = Signal(str)

    def __init__(self, source_video, source_frames, alpha_frames, output_dir, result_root,
                 source_fps, target_fps, max_edge, alpha_cutoff, alpha_source):
        super().__init__()
        self.source_video = Path(source_video) if source_video else None
        self.source_frames = {int(index): Path(path) for index, path in source_frames.items()}
        self.alpha_frames = {int(index): Path(path) for index, path in alpha_frames.items()}
        self.output_dir = Path(output_dir); self.result_root = Path(result_root)
        self.source_fps = max(0.001, float(source_fps)); self.target_fps = max(0.1, float(target_fps))
        self.max_edge = max(0, int(max_edge)); self.alpha_cutoff = max(0, min(255, int(alpha_cutoff)))
        self.alpha_source = alpha_source

    def run(self):
        try: self._run_impl()
        except Exception: self.failed.emit(traceback.format_exc())

    def _run_impl(self):
        available = sorted(index for index, path in self.alpha_frames.items() if path.is_file())
        if not available:
            raise FileNotFoundError("没有可用于 OOOSplat 导出的 Alpha / Mask 帧。")
        target_fps = min(self.source_fps, self.target_fps)
        interval = max(1.0, self.source_fps / target_fps)
        selected = []
        next_position = 0.0
        for position, source_index in enumerate(available):
            if position + 1e-6 >= next_position:
                selected.append(source_index)
                next_position += interval
        if not selected:
            selected = [available[0]]

        images_dir = self.output_dir / "images"
        alpha_dir = self.output_dir / "alpha"
        masks_dir = self.output_dir / "masks"
        for path in (images_dir, alpha_dir, masks_dir):
            path.mkdir(parents=True, exist_ok=False)

        self.progress.emit(2, f"准备 {len(selected)} 帧 OOOSplat 图片序列…")
        selected_set = set(selected); exported = []
        if self.source_video is not None and self.source_video.is_file():
            cap = cv2.VideoCapture(str(self.source_video))
            if not cap.isOpened():
                raise RuntimeError(f"无法打开源视频: {self.source_video}")
            source_index = 0
            while source_index <= selected[-1]:
                ok, frame = cap.read()
                if not ok:
                    break
                if source_index in selected_set:
                    self._write_frame(frame, source_index, len(exported), images_dir, alpha_dir, masks_dir)
                    exported.append(source_index)
                    self.progress.emit(5 + int(90 * len(exported) / len(selected)), f"生成 OOOSplat 输入 {len(exported)}/{len(selected)}")
                source_index += 1
            cap.release()
        else:
            for source_index in selected:
                source_path = self.source_frames.get(source_index)
                frame = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED) if source_path else None
                if frame is None:
                    raise FileNotFoundError(f"缺少回退源帧: {source_index:06d}.png")
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                self._write_frame(frame, source_index, len(exported), images_dir, alpha_dir, masks_dir)
                exported.append(source_index)
                self.progress.emit(5 + int(90 * len(exported) / len(selected)), f"生成 OOOSplat 输入 {len(exported)}/{len(selected)}")

        if len(exported) != len(selected):
            raise RuntimeError(f"源视频提前结束，仅导出 {len(exported)}/{len(selected)} 帧。")
        first_image = cv2.imread(str(images_dir / "000000.png"), cv2.IMREAD_UNCHANGED)
        height, width = first_image.shape[:2]
        manifest = {
            "format_version": 1,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "ooosplat_input_type": "images",
            "ooosplat_select_path": str(images_dir.resolve()),
            "source_result": str(self.result_root.resolve()),
            "source_video": str(self.source_video.resolve()) if self.source_video and self.source_video.is_file() else None,
            "alpha_source": self.alpha_source,
            "source_fps": self.source_fps,
            "sample_fps": target_fps,
            "source_frame_count": len(available),
            "exported_frame_count": len(exported),
            "width": width,
            "height": height,
            "max_edge": self.max_edge or None,
            "alpha_cutoff": self.alpha_cutoff,
            "frames": [
                {"file": f"{output_index:06d}.png", "source_frame": source_index,
                 "time_seconds": round(source_index / self.source_fps, 6)}
                for output_index, source_index in enumerate(exported)
            ],
        }
        self.progress.emit(98, "写入 OOOSplat 输入清单…")
        (self.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self.progress.emit(100, "OOOSplat 输入已生成"); self.finished.emit(str(self.output_dir))

    def _write_frame(self, frame, source_index, output_index, images_dir, alpha_dir, masks_dir):
        alpha_path = self.alpha_frames[source_index]
        alpha = cv2.imread(str(alpha_path), cv2.IMREAD_GRAYSCALE)
        if alpha is None:
            raise FileNotFoundError(f"无法读取 Alpha / Mask: {alpha_path}")
        if alpha.shape != frame.shape[:2]:
            interpolation = cv2.INTER_LINEAR if self.alpha_source == "Matting Alpha" else cv2.INTER_NEAREST
            alpha = cv2.resize(alpha, (frame.shape[1], frame.shape[0]), interpolation=interpolation)
        if self.max_edge and max(frame.shape[:2]) > self.max_edge:
            scale = self.max_edge / max(frame.shape[:2])
            size = (max(1, int(round(frame.shape[1] * scale))), max(1, int(round(frame.shape[0] * scale))))
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            alpha_interpolation = cv2.INTER_AREA if self.alpha_source == "Matting Alpha" else cv2.INTER_NEAREST
            alpha = cv2.resize(alpha, size, interpolation=alpha_interpolation)
        alpha[alpha < self.alpha_cutoff] = 0
        rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA); rgba[:, :, 3] = alpha; rgba[alpha == 0, :3] = 0
        mask = (alpha >= max(1, self.alpha_cutoff)).astype(np.uint8) * 255
        filename = f"{output_index:06d}.png"
        if not cv2.imwrite(str(images_dir / filename), rgba):
            raise OSError(f"无法写入透明图片: {filename}")
        if not cv2.imwrite(str(alpha_dir / filename), alpha):
            raise OSError(f"无法写入 Alpha: {filename}")
        if not cv2.imwrite(str(masks_dir / filename), mask):
            raise OSError(f"无法写入 Mask: {filename}")


class MainWindow(QMainWindow):
    matting_requested = Signal(dict)

    def __init__(self):
        super().__init__(); self.setWindowTitle("SAM3 / OOOSplat · Video Matte Workbench"); self.resize(1480, 900)
        self.video_path = None; self.frame_paths = {}; self.first_frame = None; self.current_frame = None; self.points = []; self.masks = {}; self.matting_masks = {}; self.output_dir = None; self.frame_count = 0; self.fps = 24.0; self.thread = None; self.worker = None; self.matting_thread = None; self.matting_worker = None; self.ooosplat_thread = None; self.ooosplat_worker = None; self.operation_started_at = None; self.operation_name = "IDLE"; self.operation_progress = 0; self.pending_seek_index = None; self.seek_timer = QTimer(self); self.seek_timer.setSingleShot(True); self.seek_timer.setInterval(40); self.seek_timer.timeout.connect(self._flush_seek); self.operation_timer = QTimer(self); self.operation_timer.setInterval(1000); self.operation_timer.timeout.connect(self._refresh_progress_meta); self.play_timer = QTimer(self); self.play_timer.timeout.connect(self._advance_frame)
        root = QWidget(); root.setObjectName("root"); self.setCentralWidget(root); outer = QVBoxLayout(root); outer.setContentsMargins(12, 10, 12, 10); outer.setSpacing(8); self.header_widget = self._build_header(); self.workspace_widget = self._build_workspace(); self.timeline_widget = self._build_timeline(); outer.addWidget(self.header_widget); outer.addWidget(self.workspace_widget, 1); outer.addWidget(self.timeline_widget); self.setStyleSheet(self._style())

    def _build_header(self):
        layout = QHBoxLayout(); brand = QLabel("OOOSPLAT / SAM3"); brand.setObjectName("brand"); project = QLabel("VIDEO MATTE WORKBENCH"); project.setObjectName("projectLabel"); self.source_label = QLabel("未加载视频"); self.source_label.setObjectName("mutedLabel"); self.state_label = QLabel("● READY"); self.state_label.setObjectName("stateReady"); self.open_button = QPushButton("打开视频"); self.open_result_button = QPushButton("打开成果"); self.output_button = QPushButton("导出设置"); self.open_button.clicked.connect(self.open_video); self.open_result_button.clicked.connect(self.open_result); self.output_button.clicked.connect(self.choose_output_root)
        layout.addWidget(brand); layout.addWidget(project); layout.addSpacing(18); layout.addWidget(self.source_label, 1); layout.addWidget(self.state_label); layout.addWidget(self.open_button); layout.addWidget(self.open_result_button); layout.addWidget(self.output_button); widget = QFrame(); widget.setLayout(layout); return widget

    def _build_workspace(self):
        layout = QHBoxLayout(); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(8); layout.addWidget(self._build_flow_panel()); layout.addWidget(self._build_preview_panel(), 1); layout.addWidget(self._build_info_panel()); widget = QWidget(); widget.setLayout(layout); return widget

    def _build_flow_panel(self):
        panel = QFrame(); panel.setObjectName("panel"); panel.setFixedWidth(210); layout = QVBoxLayout(panel); layout.addWidget(QLabel("PROCESS FLOW", objectName="panelTitle")); self.flow_items = []
        for number, label in (("01", "VIDEO SOURCE"), ("02", "SAM3 TRACK"), ("03", "MATTING"), ("04", "EXPORT")):
            item = QLabel(f"{number}   {label}"); item.setObjectName("flowActive" if number == "01" else "flowItem"); layout.addWidget(item); self.flow_items.append(item)
        layout.addSpacing(16); line = QFrame(); line.setFrameShape(QFrame.Shape.HLine); line.setObjectName("separator"); layout.addWidget(line); layout.addWidget(QLabel("OUTPUT PACKAGE", objectName="panelTitle")); self.output_edit = QLineEdit(); self.output_edit.setPlaceholderText("父目录"); layout.addWidget(self.output_edit); choose = QPushButton("选择目录"); choose.clicked.connect(self.choose_output_root); layout.addWidget(choose); layout.addStretch(1); self.run_button = QPushButton("运行 SAM3 跟踪"); self.run_button.setObjectName("primaryButton"); self.run_button.clicked.connect(self.start_tracking); layout.addWidget(self.run_button); self.clear_button = QPushButton("清除首帧提示"); self.clear_button.clicked.connect(self.clear_points); layout.addWidget(self.clear_button); return panel

    def _build_preview_panel(self):
        panel = QFrame(); panel.setObjectName("panel"); layout = QVBoxLayout(panel); toolbar = QHBoxLayout(); toolbar.addWidget(QLabel("VIEWPORT", objectName="panelTitle")); toolbar.addStretch(1); self.mode_buttons = []
        for mode, label in (("original", "原图"), ("mask", "Mask"), ("alpha", "Alpha"), ("overlay", "叠加")):
            button = QPushButton(label); button.setCheckable(True); button.clicked.connect(lambda checked, current=mode: self.set_preview_mode(current)); toolbar.addWidget(button); self.mode_buttons.append((mode, button))
        self.mode_buttons[0][1].setChecked(True); layout.addLayout(toolbar); self.preview = PreviewCanvas(); self.preview.point_added.connect(self.add_point); layout.addWidget(self.preview, 1); layout.addWidget(QLabel("左键添加前景点 · 右键添加背景点 · 处理前可持续修正首帧提示", objectName="hintLabel")); return panel

    def _build_info_panel(self):
        panel = QFrame(); panel.setObjectName("panel"); panel.setFixedWidth(250); layout = QVBoxLayout(panel); layout.addWidget(QLabel("CURRENT FRAME", objectName="panelTitle")); self.frame_info = QLabel("FRAME 000000 / 000000\nFPS --   SIZE --\nMASK NOT AVAILABLE", objectName="metricLabel"); layout.addWidget(self.frame_info); separator = QFrame(); separator.setFrameShape(QFrame.Shape.HLine); separator.setObjectName("separator"); layout.addWidget(separator); layout.addWidget(QLabel("SAM3 PARAMETERS", objectName="panelTitle")); self.prompt_info = QLabel("Foreground points  0\nBackground points  0\nTracker  READY", objectName="metricLabel"); layout.addWidget(self.prompt_info); separator2 = QFrame(); separator2.setFrameShape(QFrame.Shape.HLine); separator2.setObjectName("separator"); layout.addWidget(separator2); layout.addWidget(QLabel("PROCESS SETTINGS", objectName="panelTitle"))
        matting = QWidget(); matting_layout = QGridLayout(matting); matting_layout.setContentsMargins(7, 8, 7, 7); self.matting_model = QComboBox(); self.matting_model.addItems(["Matting Anything · MAM ViT-B"]); self.matting_checkpoint = QLineEdit(str(MATTING_CHECKPOINT)); self.matting_checkpoint.setToolTip(str(MATTING_CHECKPOINT)); self.matting_checkpoint_button = QPushButton("选择"); self.matting_checkpoint_button.clicked.connect(self.choose_matting_checkpoint); self.edge_spin = QDoubleSpinBox(); self.edge_spin.setRange(1, 32); self.edge_spin.setValue(4); self.edge_spin.setSuffix(" px"); self.matting_button = QPushButton("运行 Matting"); self.matting_button.clicked.connect(self.start_matting); matting_layout.addWidget(QLabel("Model"), 0, 0); matting_layout.addWidget(self.matting_model, 0, 1, 1, 2); matting_layout.addWidget(QLabel("Weights"), 1, 0); matting_layout.addWidget(self.matting_checkpoint, 1, 1); matting_layout.addWidget(self.matting_checkpoint_button, 1, 2); matting_layout.addWidget(QLabel("Edge"), 2, 0); matting_layout.addWidget(self.edge_spin, 2, 1); matting_layout.addWidget(self.matting_button, 3, 0, 1, 3)
        ooosplat = QWidget(); ooosplat_layout = QGridLayout(ooosplat); ooosplat_layout.setContentsMargins(7, 8, 7, 7); self.ooosplat_alpha_source = QComboBox(); self.ooosplat_alpha_source.addItems(["自动 · 优先 Matting", "Matting Alpha", "SAM3 Mask"]); self.ooosplat_fps = QDoubleSpinBox(); self.ooosplat_fps.setRange(1.0, 60.0); self.ooosplat_fps.setDecimals(1); self.ooosplat_fps.setValue(8.0); self.ooosplat_fps.setSuffix(" fps"); self.ooosplat_max_edge = QComboBox(); self.ooosplat_max_edge.addItem("原始分辨率", 0); self.ooosplat_max_edge.addItem("3840 px", 3840); self.ooosplat_max_edge.addItem("3200 px", 3200); self.ooosplat_max_edge.addItem("1920 px", 1920); self.ooosplat_max_edge.addItem("1600 px", 1600); self.ooosplat_cutoff = QSpinBox(); self.ooosplat_cutoff.setRange(0, 254); self.ooosplat_cutoff.setValue(8); self.ooosplat_cutoff.setSuffix(" / 255"); self.ooosplat_export_button = QPushButton("生成 OOOSplat 输入"); self.ooosplat_export_button.clicked.connect(self.start_ooosplat_export); self.ooosplat_export_button.setToolTip("生成可直接在 OOOSplat 中选择的透明 PNG 图片序列"); ooosplat_layout.addWidget(QLabel("Alpha"), 0, 0); ooosplat_layout.addWidget(self.ooosplat_alpha_source, 0, 1); ooosplat_layout.addWidget(QLabel("Sample"), 1, 0); ooosplat_layout.addWidget(self.ooosplat_fps, 1, 1); ooosplat_layout.addWidget(QLabel("Max edge"), 2, 0); ooosplat_layout.addWidget(self.ooosplat_max_edge, 2, 1); ooosplat_layout.addWidget(QLabel("Cutoff"), 3, 0); ooosplat_layout.addWidget(self.ooosplat_cutoff, 3, 1); ooosplat_layout.addWidget(self.ooosplat_export_button, 4, 0, 1, 2); self.process_tabs = QTabWidget(); self.process_tabs.addTab(matting, "MATTING"); self.process_tabs.addTab(ooosplat, "OOOSPLAT"); self.process_tabs.setMaximumHeight(236); layout.addWidget(self.process_tabs); self.enable_matting_controls(False); self.enable_ooosplat_controls(False)
        layout.addWidget(QLabel("EXPORT FORMAT", objectName="panelTitle")); self.export_format = QComboBox(); self.export_format.addItems(["RGBA WebM / MOV + PNG Alpha", "PNG Alpha + COLMAP Mask", "Overlay MP4"]); layout.addWidget(self.export_format); layout.addStretch(1); layout.addWidget(QLabel("OUTPUT PACKAGE", objectName="panelTitle")); self.package_list = QListWidget(); self.package_list.setMaximumHeight(115); layout.addWidget(self.package_list); return panel

    def _build_timeline(self):
        panel = QFrame(); panel.setObjectName("timelinePanel"); layout = QVBoxLayout(panel); layout.setContentsMargins(10, 6, 10, 8); layout.setSpacing(5); self.timeline_strip = TimelineStrip(); self.timeline_strip.frame_selected.connect(lambda index: self.frame_slider.setValue(index)); layout.addWidget(self.timeline_strip); row = QHBoxLayout(); self.play_button = QToolButton(); self.play_button.setText("▶"); self.play_button.setToolTip("播放 / 暂停预览"); self.play_button.clicked.connect(self.toggle_play); self.frame_slider = SeekSlider(Qt.Orientation.Horizontal); self.frame_slider.setRange(0, 0); self.frame_slider.valueChanged.connect(self._queue_seek); self.frame_slider.sliderReleased.connect(self._flush_seek); self.frame_number = QLabel("000000", objectName="monoLabel"); self.preview_source = QComboBox(); self.preview_source.addItems(["SAM3 Mask"]); self.preview_source.setEnabled(False); self.preview_source.currentIndexChanged.connect(lambda _: self.seek_frame(self.frame_slider.value())); self.preview_source.setToolTip("选择播放 SAM3 或 Matting 阶段的 Alpha 结果"); row.addWidget(self.play_button); row.addWidget(self.frame_slider, 1); row.addWidget(self.frame_number); row.addWidget(self.preview_source); layout.addLayout(row)
        monitor = QFrame(); monitor.setObjectName("progressMonitor"); monitor_layout = QVBoxLayout(monitor); monitor_layout.setContentsMargins(9, 6, 9, 6); monitor_layout.setSpacing(4); info_row = QHBoxLayout(); info_row.setSpacing(10); self.progress_stage = QLabel("IDLE", objectName="progressStage"); self.status = QLabel("等待视频", objectName="progressStatus"); self.progress_meta = QLabel("--:--  ·  ETA --:--", objectName="progressMeta"); self.progress_percent = QLabel("000%", objectName="progressPercent"); info_row.addWidget(self.progress_stage); info_row.addWidget(self.status, 1); info_row.addWidget(self.progress_meta); info_row.addWidget(self.progress_percent); monitor_layout.addLayout(info_row); self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0); self.progress.setTextVisible(False); self.progress.setFixedHeight(10); monitor_layout.addWidget(self.progress); scale = QHBoxLayout(); scale.setContentsMargins(1, 0, 1, 0); scale.setSpacing(0)
        for label in ("INIT", "INFERENCE", "ENCODE", "DONE"):
            marker = QLabel(label, objectName="progressMarker"); scale.addWidget(marker); scale.addStretch(1)
        scale.takeAt(scale.count() - 1); monitor_layout.addLayout(scale); layout.addWidget(monitor); return panel

    def _style(self):
        ui_font = preferred_font(["IBM Plex Sans", "Inter", "Microsoft YaHei UI"], "Segoe UI")
        mono_font = preferred_font(["JetBrains Mono", "Cascadia Mono", "Consolas"], "Microsoft YaHei UI")
        return f"""* {{ font-family: '{ui_font}'; color: {COLORS['text']}; font-size: 12px; }} QMainWindow, QWidget#root {{ background: {COLORS['bg']}; }} QFrame#panel, QFrame#timelinePanel {{ background: {COLORS['panel']}; border: 1px solid {COLORS['line']}; border-radius: 8px; }} QFrame#previewCanvas {{ background: #0D1116; border: 1px solid {COLORS['line']}; border-radius: 6px; }} QFrame#timelineStrip {{ background: #131A22; border: 1px solid {COLORS['line']}; border-radius: 5px; }} QFrame#progressMonitor {{ background: #151C24; border: 1px solid #2B3742; border-radius: 6px; }} QFrame#separator {{ color: {COLORS['line']}; background: {COLORS['line']}; max-height: 1px; border: 0; }} QLabel#brand {{ color: {COLORS['teal']}; font-family: '{mono_font}'; font-size: 15px; font-weight: 800; }} QLabel#projectLabel, QLabel#panelTitle {{ color: {COLORS['muted']}; font-family: '{mono_font}'; font-size: 10px; letter-spacing: 1px; }} QLabel#mutedLabel, QLabel#hintLabel {{ color: {COLORS['muted']}; }} QLabel#stateReady {{ color: {COLORS['teal']}; font-family: '{mono_font}'; font-weight: 700; }} QLabel#metricLabel, QLabel#monoLabel {{ font-family: '{mono_font}'; color: {COLORS['muted']}; }} QLabel#progressStage {{ color: {COLORS['teal']}; font-family: '{mono_font}'; font-size: 11px; font-weight: 800; min-width: 142px; }} QLabel#progressStatus {{ color: #C8D3DA; }} QLabel#progressMeta {{ color: {COLORS['muted']}; font-family: '{mono_font}'; min-width: 145px; }} QLabel#progressPercent {{ color: {COLORS['teal']}; font-family: '{mono_font}'; font-size: 13px; font-weight: 800; min-width: 42px; }} QLabel#progressMarker {{ color: #647381; font-family: '{mono_font}'; font-size: 9px; }} QLabel#progressStage[state="error"], QLabel#progressPercent[state="error"] {{ color: {COLORS['red']}; }} QLabel#flowActive {{ color: {COLORS['teal']}; background: {COLORS['teal_dim']}; padding: 10px 8px; border-left: 3px solid {COLORS['teal']}; font-family: '{mono_font}'; }} QLabel#flowItem {{ color: {COLORS['muted']}; padding: 10px 8px; font-family: '{mono_font}'; }} QPushButton, QToolButton {{ background: {COLORS['panel_2']}; border: 1px solid {COLORS['line']}; border-radius: 6px; padding: 7px 11px; }} QPushButton:hover, QToolButton:hover {{ border-color: {COLORS['teal']}; color: {COLORS['teal']}; }} QPushButton:checked {{ background: {COLORS['teal_dim']}; color: {COLORS['teal']}; border-color: {COLORS['teal']}; }} QPushButton#primaryButton {{ background: {COLORS['teal']}; color: #0C1618; border: 0; font-weight: 800; padding: 10px; }} QPushButton:disabled, QComboBox:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled {{ color: #5E6974; background: #171D25; border-color: #29333E; }} QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {{ background: #131A22; border: 1px solid {COLORS['line']}; border-radius: 5px; padding: 7px; }} QGroupBox {{ border: 1px solid {COLORS['line']}; border-radius: 6px; margin-top: 10px; padding: 8px; color: {COLORS['muted']}; }} QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }} QListWidget {{ background: #131A22; border: 1px solid {COLORS['line']}; border-radius: 5px; color: {COLORS['muted']}; }} QTabWidget::pane {{ background: #151C24; border: 1px solid {COLORS['line']}; border-radius: 5px; }} QTabBar::tab {{ background: #171F28; color: {COLORS['muted']}; border: 1px solid {COLORS['line']}; border-bottom: 0; padding: 6px 11px; }} QTabBar::tab:selected {{ background: {COLORS['teal_dim']}; color: {COLORS['teal']}; border-color: {COLORS['teal']}; }} QSlider::groove:horizontal {{ height: 4px; background: #35414D; }} QSlider::handle:horizontal {{ width: 12px; margin: -5px 0; background: {COLORS['teal']}; border-radius: 6px; }} QProgressBar {{ background: #26313C; border: 1px solid #34414E; border-radius: 4px; }} QProgressBar::chunk {{ background: {COLORS['teal']}; border-radius: 3px; }} QProgressBar[state="error"]::chunk {{ background: {COLORS['red']}; }} QMessageBox {{ background-color: {COLORS['panel']}; }} QMessageBox QLabel, QMessageBox QLabel#qt_msgbox_label {{ color: {COLORS['text']}; background: transparent; min-width: 360px; padding: 8px; }} QMessageBox QPushButton {{ min-width: 72px; color: {COLORS['text']}; background: {COLORS['panel_2']}; border: 1px solid {COLORS['line']}; }} QMessageBox QPushButton:hover {{ color: {COLORS['teal']}; border-color: {COLORS['teal']}; }}"""

    @staticmethod
    def _format_duration(seconds):
        total = max(0, int(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"

    def _set_progress_visual_state(self, state):
        for widget in (self.progress, self.progress_stage, self.progress_percent):
            widget.setProperty("state", state)
            widget.style().unpolish(widget)
            widget.style().polish(widget)

    def _begin_progress(self, operation_name, message):
        self.operation_name = operation_name.upper()
        self.operation_started_at = time.monotonic()
        self.operation_progress = 0
        self.progress.setValue(0)
        self.progress_percent.setText("000%")
        self.progress_stage.setText(f"{self.operation_name} · INIT")
        self.status.setText(message)
        self._set_progress_visual_state("running")
        self._refresh_progress_meta()
        self.operation_timer.start()

    def _update_progress(self, value, message):
        value = max(0, min(100, int(value)))
        self.operation_progress = value
        if value >= 100:
            phase = "DONE"
        elif value >= 89 or any(word in message for word in ("导出", "编码", "写入")):
            phase = "ENCODE"
        elif value <= 4 or "加载" in message:
            phase = "INIT"
        else:
            phase = "INFERENCE"
        self.progress.setValue(value)
        self.progress_percent.setText(f"{value:03d}%")
        self.progress_stage.setText(f"{self.operation_name} · {phase}")
        self.status.setText(message)
        self._refresh_progress_meta()

    def _refresh_progress_meta(self):
        if self.operation_started_at is None:
            return
        elapsed = max(0.0, time.monotonic() - self.operation_started_at)
        if 5 <= self.operation_progress < 100:
            eta = elapsed * (100 - self.operation_progress) / self.operation_progress
            eta_text = self._format_duration(eta)
        elif self.operation_progress >= 100:
            eta_text = "00:00"
        else:
            eta_text = "--:--"
        self.progress_meta.setText(f"{self._format_duration(elapsed)}  ·  ETA {eta_text}")

    def _finish_progress(self, message):
        self._update_progress(100, message)
        self.operation_timer.stop()
        self._set_progress_visual_state("complete")

    def _fail_progress(self, message):
        self.operation_timer.stop()
        self.progress_stage.setText(f"{self.operation_name} · ERROR")
        self.status.setText(message)
        self._refresh_progress_meta()
        self._set_progress_visual_state("error")

    def _reset_progress(self, message="等待视频"):
        self.operation_timer.stop()
        self.operation_started_at = None
        self.operation_name = "IDLE"
        self.operation_progress = 0
        self.progress.setValue(0)
        self.progress_stage.setText("IDLE")
        self.status.setText(message)
        self.progress_meta.setText("--:--  ·  ETA --:--")
        self.progress_percent.setText("000%")
        self._set_progress_visual_state("idle")

    def _show_loaded_progress(self, message):
        self.operation_timer.stop()
        self.operation_started_at = None
        self.operation_name = "RESULT"
        self.operation_progress = 100
        self.progress.setValue(100)
        self.progress_stage.setText("RESULT · LOADED")
        self.status.setText(message)
        self.progress_meta.setText("已完成  ·  可预览")
        self.progress_percent.setText("100%")
        self._set_progress_visual_state("complete")

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video (*.mp4 *.mov *.mkv *.avi *.webm)")
        if not path: return
        cap = cv2.VideoCapture(path); ok, frame = cap.read(); count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 24.0; cap.release()
        if not ok: QMessageBox.critical(self, "打开失败", "无法读取视频首帧。"); return
        self.video_path, self.frame_paths, self.first_frame, self.current_frame, self.frame_count, self.fps = path, {}, frame, frame, count, fps; self.masks.clear(); self.matting_masks.clear(); self.enable_matting_controls(False); self.enable_ooosplat_controls(False); self.preview_source.clear(); self.preview_source.addItem("SAM3 Mask"); self.preview_source.setEnabled(False); self.points.clear(); self.preview.set_points([]); self.preview.set_frame(frame); self.timeline_strip.set_video(path, count); self.frame_slider.setRange(0, max(0, count - 1)); self.frame_slider.setValue(0); self.source_label.setText(f"SOURCE  {Path(path).name}  ·  {count} frames  ·  {fps:.2f} fps"); self.output_edit.setText(self.output_edit.text().strip() or str(Path(path).parent)); self._update_frame_info(); self._update_prompt_info(); self._set_state("READY", COLORS["teal"]); self._set_flow_step(0); self._reset_progress("READY · 首帧已载入，添加主体提示")

    def open_result(self):
        start = self.output_edit.text().strip() or str(APP_DIR)
        selected = QFileDialog.getExistingDirectory(self, "选择 SAM3 / Matting 成果文件夹", start)
        if not selected: return
        try: self._load_result_package(Path(selected))
        except Exception as exc: QMessageBox.critical(self, "成果读取失败", str(exc))

    def _load_result_package(self, selected):
        root = selected.parent if selected.name.startswith("matting_output") and (selected.parent / "metadata.json").exists() else selected
        metadata_path = root / "metadata.json"
        if not metadata_path.exists(): raise FileNotFoundError("所选文件夹中没有 metadata.json，请选择完整的 *_sam3_output 成果文件夹。")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")); source_text = metadata.get("source_video", ""); source = Path(source_text) if source_text else None; fallback_dir = root / "rgba_frames"; fallback_paths = sorted(fallback_dir.glob("*.png"))
        if source is not None and source.is_file():
            cap = cv2.VideoCapture(str(source)); ok, frame = cap.read(); count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(metadata.get("frame_count", 0)); fps = cap.get(cv2.CAP_PROP_FPS) or float(metadata.get("fps", 24.0)); cap.release(); self.video_path = str(source); self.frame_paths = {}; self.timeline_strip.set_video(str(source), count)
        elif fallback_paths:
            frame = cv2.imread(str(fallback_paths[0]), cv2.IMREAD_COLOR); ok = frame is not None; count = len(fallback_paths); fps = float(metadata.get("fps", 24.0)); self.video_path = None; self.frame_paths = {int(path.stem): path for path in fallback_paths}; self.timeline_strip.set_frames(fallback_paths)
        else: raise FileNotFoundError(f"原视频不存在，且成果中没有可回退预览的 rgba_frames：\n{source_text or '未记录'}")
        if not ok: raise RuntimeError("无法读取成果的首帧。")
        self.output_dir = root; self.first_frame = frame; self.current_frame = frame; self.frame_count = count; self.fps = fps; self.masks = {int(path.stem): path for path in sorted((root / "mask_frames").glob("*.png"))}
        matting_dirs = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("matting_output") and (path / "matting_frames").exists()]; matting_dir = max(matting_dirs, key=lambda path: path.stat().st_mtime) if matting_dirs else None; self.matting_masks = {int(path.stem): path for path in sorted((matting_dir / "matting_frames").glob("*.png"))} if matting_dir else {}
        global ACTIVE_OUTPUT_DIR
        ACTIVE_OUTPUT_DIR = root; self.points = [(float(item["x"]), float(item["y"]), int(bool(item.get("positive", True)))) for item in metadata.get("points", [])]; self.preview.set_points(self.points); self.preview_source.clear(); self.preview_source.addItem("SAM3 Mask")
        if self.matting_masks: self.preview_source.addItem("Matting Alpha")
        self.preview_source.setEnabled(bool(self.matting_masks)); self.preview_source.setCurrentIndex(0); self.timeline_strip.set_mask_frames(self.masks.keys()); self.frame_slider.setRange(0, max(0, count - 1)); self.frame_slider.setValue(0); self.preview.set_frame(frame, None); self.seek_frame(0); self.output_edit.setText(str(root.parent)); self.source_label.setText(f"RESULT  {root.name}  ·  {count} frames  ·  {fps:.2f} fps"); self.package_list.clear(); self.package_list.addItems([path.name for path in sorted(root.iterdir())]); self.enable_matting_controls(bool(self.masks) and self.video_path is not None); self.enable_ooosplat_controls(bool(self.masks) and (bool(self.video_path) or bool(self.frame_paths))); self._update_prompt_info(); self._set_state("LOADED", COLORS["teal"]); self._set_flow_step(3 if self.matting_masks else 2); fallback_note = " · 原视频缺失，使用 RGBA 帧预览" if self.video_path is None else ""; self._show_loaded_progress(f"LOADED · 已读取成果 {root}{fallback_note}")

    def choose_output_root(self):
        start = self.output_edit.text().strip() or (str(Path(self.video_path).parent) if self.video_path else ""); path = QFileDialog.getExistingDirectory(self, "选择结果保存父目录", start)
        if path: self.output_edit.setText(path)

    def add_point(self, x, y, positive):
        self.points.append((x, y, int(positive))); self.preview.set_points(self.points); self._update_prompt_info(); self.status.setText(f"PROMPT · {'前景' if positive else '背景'}点已添加 · 可继续添加")

    def clear_points(self):
        self.points.clear(); self.preview.set_points([]); self._update_prompt_info(); self.status.setText("PROMPT · 已清除")

    def set_preview_mode(self, mode):
        for current, button in self.mode_buttons: button.setChecked(current == mode)
        self.preview.set_mode(mode)

    def _update_frame_info(self):
        mask_map = self.matting_masks if self.preview_source.currentIndex() == 1 else self.masks; mask_state = ("MATTING ALPHA READY" if self.preview_source.currentIndex() == 1 else "SAM3 MASK READY") if self.frame_slider.value() in mask_map else "MASK NOT AVAILABLE"; size = "--" if self.current_frame is None else f"{self.current_frame.shape[1]}×{self.current_frame.shape[0]}"; self.frame_info.setText(f"FRAME {self.frame_slider.value():06d} / {max(0, self.frame_count - 1):06d}\nFPS {self.fps:.2f}   SIZE {size}\n{mask_state}")

    def _update_prompt_info(self):
        fg = sum(1 for _, _, p in self.points if p); self.prompt_info.setText(f"Foreground points  {fg}\nBackground points  {len(self.points) - fg}\nTracker  {'COMPLETE' if self.masks else 'READY'}")

    def _queue_seek(self, index):
        self.pending_seek_index = int(index)
        self.frame_number.setText(f"{int(index):06d}")
        self.timeline_strip.set_current(int(index))
        if not self.frame_slider.isSliderDown():
            self._flush_seek()
        elif not self.seek_timer.isActive():
            self.seek_timer.start()

    def _flush_seek(self):
        self.seek_timer.stop()
        if self.pending_seek_index is None:
            return
        index = self.pending_seek_index
        self.pending_seek_index = None
        self.seek_frame(index)

    def seek_frame(self, index):
        if not self.video_path and not self.frame_paths: return
        if self.video_path:
            cap = cv2.VideoCapture(self.video_path); cap.set(cv2.CAP_PROP_POS_FRAMES, index); ok, frame = cap.read(); cap.release()
        else:
            frame = cv2.imread(str(self.frame_paths.get(index)), cv2.IMREAD_COLOR) if index in self.frame_paths else None; ok = frame is not None
        if ok:
            mask_map = self.matting_masks if self.preview_source.currentIndex() == 1 else self.masks; mask_path = mask_map.get(index); mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0) if self.preview_source.currentIndex() == 1 and isinstance(mask_path, Path) else (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127 if isinstance(mask_path, Path) else mask_path)
            self.current_frame = frame; self.preview.set_frame(frame, mask); self.timeline_strip.set_current(index); self.frame_number.setText(f"{index:06d}"); self._update_frame_info()

    def toggle_play(self):
        if self.play_timer.isActive(): self.play_timer.stop(); self.play_button.setText("▶")
        else: self.play_timer.start(max(15, int(1000 / self.fps))); self.play_button.setText("Ⅱ")

    def _advance_frame(self):
        if self.frame_slider.value() >= self.frame_slider.maximum(): self.play_timer.stop(); self.play_button.setText("▶"); return
        self.frame_slider.setValue(self.frame_slider.value() + 1)

    def enable_matting_controls(self, enabled):
        self.matting_model.setEnabled(enabled); self.matting_checkpoint.setEnabled(enabled); self.matting_checkpoint_button.setEnabled(enabled); self.edge_spin.setEnabled(enabled); self.matting_button.setEnabled(enabled)

    def enable_ooosplat_controls(self, enabled):
        self.ooosplat_alpha_source.setEnabled(enabled); self.ooosplat_fps.setEnabled(enabled); self.ooosplat_max_edge.setEnabled(enabled); self.ooosplat_cutoff.setEnabled(enabled); self.ooosplat_export_button.setEnabled(enabled)

    def matting_settings(self):
        return {"model": self.matting_model.currentText(), "edge_width": self.edge_spin.value(), "checkpoint": self.matting_checkpoint.text().strip(), "input_dir": str(self.output_dir / "mask_frames") if self.output_dir else None, "output_dir": str(self.output_dir / "matting_output") if self.output_dir else None}

    def _emit_matting_request(self):
        self.matting_requested.emit(self.matting_settings())

    def choose_matting_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 Matting Anything 权重", str(Path(self.matting_checkpoint.text()).parent), "PyTorch checkpoint (*.pth *.pt)")
        if path: self.matting_checkpoint.setText(path)

    def start_matting(self):
        if not self.output_dir or not self.video_path or not (self.output_dir / "mask_frames").exists():
            QMessageBox.warning(self, "无法运行 Matting", "请先完成 SAM3 跟踪并生成 Mask。"); return
        checkpoint = Path(self.matting_checkpoint.text().strip() or MATTING_CHECKPOINT)
        if not checkpoint.exists():
            QMessageBox.warning(self, "缺少 Matting 权重", f"找不到 MAM 权重：\n{checkpoint}\n\n请先下载模型或选择正确的 .pth 文件。"); return
        self.matting_requested.emit(self.matting_settings()); self._matting_pending = None; matting_output = self.output_dir / "matting_output"; suffix = 2
        while matting_output.exists() and any(matting_output.iterdir()): matting_output = self.output_dir / f"matting_output_{suffix}"; suffix += 1
        self.matting_button.setEnabled(False); self.matting_checkpoint_button.setEnabled(False); self.enable_ooosplat_controls(False); self._set_state("MATTING", COLORS["amber"]); self._set_flow_step(2); self._begin_progress("MATTING", "正在准备 Matting Anything…")
        self.matting_thread = QThread(self); self.matting_worker = MattingWorker(self.video_path, self.output_dir, matting_output, checkpoint, self.edge_spin.value()); self.matting_worker.moveToThread(self.matting_thread); self.matting_thread.started.connect(self.matting_worker.run); self.matting_worker.progress.connect(self._update_progress); self.matting_worker.finished.connect(self.matting_finished); self.matting_worker.failed.connect(self.matting_failed); self.matting_worker.finished.connect(self.matting_worker.deleteLater); self.matting_worker.failed.connect(self.matting_worker.deleteLater); self.matting_worker.finished.connect(self.matting_thread.quit); self.matting_worker.failed.connect(self.matting_thread.quit); self.matting_thread.finished.connect(self._matting_thread_finished); self.matting_thread.finished.connect(self.matting_thread.deleteLater); self.matting_thread.start()

    def start_ooosplat_export(self):
        if not self.output_dir or not self.masks:
            QMessageBox.warning(self, "无法生成 OOOSplat 输入", "请先完成 SAM3 跟踪并生成 Mask。")
            return
        source_choice = self.ooosplat_alpha_source.currentText()
        if source_choice == "Matting Alpha" and not self.matting_masks:
            QMessageBox.warning(self, "缺少 Matting Alpha", "当前成果中没有 Matting Alpha，请选择自动或 SAM3 Mask。")
            return
        if source_choice == "SAM3 Mask" or not self.matting_masks:
            alpha_frames = self.masks
            alpha_source = "SAM3 Mask"
        else:
            alpha_frames = self.matting_masks
            alpha_source = "Matting Alpha"
        source_video = self.video_path if self.video_path and Path(self.video_path).is_file() else None
        if source_video is None and not self.frame_paths:
            QMessageBox.warning(self, "缺少原始画面", "原视频和回退 RGBA 帧均不可用，无法生成 OOOSplat 输入。")
            return
        export_dir = self.output_dir / "ooosplat_input"
        suffix = 2
        while export_dir.exists():
            export_dir = self.output_dir / f"ooosplat_input_{suffix}"
            suffix += 1
        self.enable_ooosplat_controls(False); self.enable_matting_controls(False); self.run_button.setEnabled(False); self.open_button.setEnabled(False); self.open_result_button.setEnabled(False); self.output_button.setEnabled(False)
        self._set_state("EXPORT", COLORS["amber"]); self._set_flow_step(3); self._begin_progress("OOOSPLAT", f"准备 {alpha_source} 图片序列…")
        self._ooosplat_pending = None
        self.ooosplat_thread = QThread(self)
        self.ooosplat_worker = OoosplatExportWorker(
            source_video, {index: str(path) for index, path in self.frame_paths.items()},
            {index: str(path) for index, path in alpha_frames.items()}, export_dir, self.output_dir,
            self.fps, self.ooosplat_fps.value(), self.ooosplat_max_edge.currentData() or 0,
            self.ooosplat_cutoff.value(), alpha_source,
        )
        self.ooosplat_worker.moveToThread(self.ooosplat_thread); self.ooosplat_thread.started.connect(self.ooosplat_worker.run); self.ooosplat_worker.progress.connect(self._update_progress); self.ooosplat_worker.finished.connect(self.ooosplat_export_finished); self.ooosplat_worker.failed.connect(self.ooosplat_export_failed); self.ooosplat_worker.finished.connect(self.ooosplat_worker.deleteLater); self.ooosplat_worker.failed.connect(self.ooosplat_worker.deleteLater); self.ooosplat_worker.finished.connect(self.ooosplat_thread.quit); self.ooosplat_worker.failed.connect(self.ooosplat_thread.quit); self.ooosplat_thread.finished.connect(self._ooosplat_thread_finished); self.ooosplat_thread.finished.connect(self.ooosplat_thread.deleteLater); self.ooosplat_thread.start()

    def ooosplat_export_finished(self, output_dir):
        output = Path(output_dir); self.package_list.clear(); self.package_list.addItems([path.name for path in sorted(self.output_dir.iterdir())]); self._set_state("COMPLETE", COLORS["teal"]); self._set_flow_step(3); self._finish_progress(f"OOOSplat 输入已写入 {output / 'images'}"); self._ooosplat_pending = ("success", f"OOOSplat 输入已生成：\n{output}\n\n在 OOOSplat 中选择图片序列目录：\n{output / 'images'}")

    def ooosplat_export_failed(self, error):
        log_path = self.output_dir / "ooosplat_export_error.log"; log_path.write_text(error, encoding="utf-8"); self._set_state("ERROR", COLORS["red"]); self._fail_progress("OOOSplat 输入生成失败 · 详情已写入日志"); self._ooosplat_pending = ("error", error)

    def _ooosplat_thread_finished(self):
        self.ooosplat_worker = None; self.ooosplat_thread = None; self.run_button.setEnabled(True); self.open_button.setEnabled(True); self.open_result_button.setEnabled(True); self.output_button.setEnabled(True); self.enable_matting_controls(bool(self.video_path)); self.enable_ooosplat_controls(bool(self.masks) and (bool(self.video_path) or bool(self.frame_paths)))
        pending = self._ooosplat_pending; self._ooosplat_pending = None
        if not pending: return
        kind, message = pending
        QTimer.singleShot(0, lambda: QMessageBox.information(self, "OOOSplat 输入完成", message) if kind == "success" else QMessageBox.critical(self, "OOOSplat 输入失败", message))

    def _set_state(self, text, color):
        self.state_label.setText(f"● {text}"); self.state_label.setStyleSheet(f"color:{color}; font-weight:700;")

    def _set_flow_step(self, active_index):
        for index, item in enumerate(self.flow_items):
            item.setObjectName("flowActive" if index == active_index else "flowItem"); item.style().unpolish(item); item.style().polish(item)

    def start_tracking(self):
        if not self.video_path or not self.points or not any(p[2] for p in self.points): QMessageBox.warning(self, "缺少主体提示", "请先打开视频，并在首帧用左键至少点击一个主体内部点。"); return
        output_root = Path(self.output_edit.text().strip()) if self.output_edit.text().strip() else Path(self.video_path).parent; output_root.mkdir(parents=True, exist_ok=True); base = Path(self.video_path).stem + "_sam3_output"; output_dir = output_root / base; suffix = 2
        while output_dir.exists(): output_dir = output_root / f"{base}_{suffix}"; suffix += 1
        global ACTIVE_OUTPUT_DIR
        self.output_dir = output_dir; ACTIVE_OUTPUT_DIR = output_dir; self.run_button.setEnabled(False); self.open_button.setEnabled(False); self.enable_matting_controls(False); self.enable_ooosplat_controls(False); self._set_state("RUNNING", COLORS["amber"]); self._set_flow_step(1); self._begin_progress("SAM3", "正在准备视频跟踪…")
        self._pending_dialog = None; self.thread = QThread(self); self.worker = TrackWorker(self.video_path, self.points.copy(), output_dir); self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.run); self.worker.progress.connect(self._update_progress); self.worker.finished.connect(self.tracking_finished); self.worker.failed.connect(self.tracking_failed); self.worker.finished.connect(self.worker.deleteLater); self.worker.failed.connect(self.worker.deleteLater); self.worker.finished.connect(self.thread.quit); self.worker.failed.connect(self.thread.quit); self.thread.finished.connect(self._tracking_thread_finished); self.thread.finished.connect(self.thread.deleteLater); self.thread.start()

    def tracking_finished(self, output_dir):
        output = Path(output_dir); self.masks.clear()
        for path in sorted((output / "mask_frames").glob("*.png")): self.masks[int(path.stem)] = path
        self.timeline_strip.set_mask_frames(self.masks.keys()); self.preview_source.setCurrentIndex(0); self.preview_source.setEnabled(False); self.package_list.clear(); self.package_list.addItems([p.name for p in sorted(output.iterdir())]); self.frame_slider.setValue(0); self.seek_frame(0); self._update_prompt_info(); self.enable_matting_controls(True); self.enable_ooosplat_controls(True); self._set_state("COMPLETE", COLORS["teal"]); self._set_flow_step(2); self._finish_progress(f"SAM3 + RGBA 输出已写入 {output} · 可接入 Matting"); metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8")); rgba_name = metadata.get("rgba_video") or "未生成（保留 RGBA PNG 序列）"; self._pending_dialog = ("success", f"结果已保存到：\n{output}\n\n透明视频：{rgba_name}\n透明帧：rgba_frames\\*.png")

    def tracking_failed(self, error):
        self._set_state("ERROR", COLORS["red"]); self._fail_progress("SAM3 处理失败 · 详情见错误提示"); self._pending_dialog = ("error", error)

    def _tracking_thread_finished(self):
        self.run_button.setEnabled(True); self.open_button.setEnabled(True)
        self.worker = None
        self.thread = None
        pending = self._pending_dialog; self._pending_dialog = None
        if not pending:
            return
        kind, message = pending
        QTimer.singleShot(0, lambda: QMessageBox.information(self, "跟踪完成", message) if kind == "success" else QMessageBox.critical(self, "SAM3 处理失败", message))

    def matting_finished(self, output_dir):
        output = Path(output_dir); self.matting_masks = {int(path.stem): path for path in sorted((output / "matting_frames").glob("*.png"))}; self.preview_source.clear(); self.preview_source.addItems(["SAM3 Mask", "Matting Alpha"]); self.preview_source.setEnabled(bool(self.matting_masks)); self.preview_source.setCurrentIndex(0); self.package_list.clear(); self.package_list.addItems([p.name for p in sorted(self.output_dir.iterdir())]); self._set_state("COMPLETE", COLORS["teal"]); self._set_flow_step(3); self._finish_progress(f"Matting Alpha 输出已写入 {output}"); metadata = json.loads((output / "matting_metadata.json").read_text(encoding="utf-8")); rgba_name = metadata.get("rgba_video") or "未生成（保留 RGBA PNG 序列）"; self._matting_pending = ("success", f"Matting 结果已保存到：\n{output}\n\n透明视频：{rgba_name}\nAlpha：matting_frames\\*.png\nRGBA：matting_rgba_frames\\*.png")

    def matting_failed(self, error):
        log_path = (self.output_dir / "sam3_ooosplat_error.log") if self.output_dir else (APP_DIR / "sam3_ooosplat_error.log"); log_path.parent.mkdir(parents=True, exist_ok=True); log_path.write_text(error, encoding="utf-8"); self._set_state("ERROR", COLORS["red"]); self._fail_progress("Matting 处理失败 · 详情已写入日志"); self._matting_pending = ("error", error)

    def _matting_thread_finished(self):
        self.matting_worker = None; self.matting_thread = None; self.enable_matting_controls(True); self.enable_ooosplat_controls(bool(self.masks))
        pending = self._matting_pending; self._matting_pending = None
        if not pending: return
        kind, message = pending
        QTimer.singleShot(0, lambda: QMessageBox.information(self, "Matting 完成", message) if kind == "success" else QMessageBox.critical(self, "Matting 处理失败", message))


if __name__ == "__main__":
    app = QApplication(sys.argv); app.setStyle("Fusion"); window = MainWindow(); window.show(); sys.exit(app.exec())
