"""Input/output utilities for RVR: frame reading, GPU preprocessing and writing.

Supports both video files (read with ``decord``) and image folders, and writes
either an mp4 (libx265) or a PNG sequence.
"""

import os
import math
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import imageio
from PIL import Image

import decord
decord.bridge.set_bridge("torch")

from .streaming.chunk import ChunkSpec, build_chunk_specs

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
KEEP_MAX_4K1 = True
CROP_LQ_MULTIPLE = 8

_INTERP_NEEDS_ALIGN = ("linear", "bilinear", "bicubic", "trilinear")


def is_video_file(filename) -> bool:
    return str(filename).lower().endswith(VIDEO_EXTS)


# --------------------------------------------------------------------------- #
# Frame listing / reading                                                     #
# --------------------------------------------------------------------------- #

def _is_valid_image_file(path: Path) -> bool:
    if not path.is_file() or path.name.startswith("."):
        return False
    return path.suffix.lower() in IMAGE_EXTS


def _numeric_sort_key(path: Path):
    try:
        return int(path.stem)
    except ValueError:
        return path.name


def _max_4k_plus_1_count(n: int) -> int:
    return ((n - 1) // 4) * 4 + 1 if n > 0 else 0


def list_image_frames(folder_path) -> List[Path]:
    folder = Path(folder_path)
    frames = sorted((p for p in folder.iterdir() if _is_valid_image_file(p)), key=_numeric_sort_key)
    if KEEP_MAX_4K1:
        frames = frames[:_max_4k_plus_1_count(len(frames))]
    return frames


def _crop_size_to_multiple(h: int, w: int, multiple: int) -> Tuple[int, int]:
    if multiple is None or multiple <= 1:
        return h, w
    crop_h, crop_w = (h // multiple) * multiple, (w // multiple) * multiple
    if crop_h <= 0 or crop_w <= 0:
        raise ValueError(f"Invalid crop: {h}x{w}, multiple={multiple}")
    return crop_h, crop_w


def _decord_batch_to_torch(frames):
    if isinstance(frames, torch.Tensor):
        return frames
    if hasattr(frames, "asnumpy"):
        return torch.from_numpy(frames.asnumpy())
    return torch.as_tensor(frames)


def _read_image_chunk_uint8(paths: List[Path], crop_h: int, crop_w: int) -> torch.Tensor:
    arrs = []
    for p in paths:
        with Image.open(p) as img:
            img = img.convert("RGB")
            w, h = img.size
            if h < crop_h or w < crop_w:
                raise ValueError(f"Frame too small: {p}, frame={h}x{w}, crop={crop_h}x{crop_w}")
            arrs.append(np.asarray(img.crop((0, 0, crop_w, crop_h)), dtype=np.uint8))
    return torch.from_numpy(np.stack(arrs, axis=0)).contiguous()


def get_video_info(video_path, fallback_fps=30) -> Tuple[int, int, int, float]:
    """Return (total_frames, lq_height, lq_width, fps) with sizes cropped to a
    multiple of ``CROP_LQ_MULTIPLE``."""
    path = Path(video_path)
    if path.is_dir():
        frames = list_image_frames(path)
        if not frames:
            raise ValueError(f"No valid image frames in: {path}")
        with Image.open(frames[0]) as img:
            w, h = img.convert("RGB").size
        crop_h, crop_w = _crop_size_to_multiple(h, w, CROP_LQ_MULTIPLE)
        return len(frames), crop_h, crop_w, float(fallback_fps)

    vr = decord.VideoReader(uri=path.as_posix())
    f0 = vr[0]
    try:
        fps = float(vr.get_avg_fps())
    except Exception:
        fps = float(fallback_fps)
    if not math.isfinite(fps) or fps <= 0:
        fps = float(fallback_fps)
    crop_h, crop_w = _crop_size_to_multiple(f0.shape[0], f0.shape[1], CROP_LQ_MULTIPLE)
    return len(vr), crop_h, crop_w, fps


def selected_output_frame_names(input_folder) -> List[str]:
    return [f"{p.stem}.png" for p in list_image_frames(Path(input_folder))]


def iter_video_clips_fixed_scheme(
    video_path, clip_len: int, total_frames: int, crop_h: int, crop_w: int,
) -> Iterator[Tuple[ChunkSpec, torch.Tensor]]:
    """Yield ``(spec, raw_uint8_frames)`` for each fixed-size chunk.

    ``raw_uint8_frames`` is a CPU ``[T, H, W, 3]`` uint8 tensor.
    """
    path = Path(video_path)
    specs = build_chunk_specs(total_frames, clip_len)

    if path.is_dir():
        all_paths = list_image_frames(path)
        for spec in specs:
            chunk_paths = all_paths[spec.frame_start: spec.frame_start + spec.frame_count]
            yield spec, _read_image_chunk_uint8(chunk_paths, crop_h, crop_w)
    else:
        try:
            decord.bridge.set_bridge("torch")
        except Exception:
            pass
        vr = decord.VideoReader(uri=path.as_posix())
        for spec in specs:
            idx = list(range(spec.frame_start, spec.frame_start + spec.frame_count))
            frames = _decord_batch_to_torch(vr.get_batch(idx))
            yield spec, frames[:, :crop_h, :crop_w, :].contiguous()


# --------------------------------------------------------------------------- #
# GPU preprocessing                                                           #
# --------------------------------------------------------------------------- #

def preprocess_clip_uint8(frames_uint8, out_h, out_w, mode, pad_h, pad_w, dtype):
    """``[T, H, W, 3]`` uint8 (CUDA) -> ``[1, T, 3, out_h+pad_h, out_w+pad_w]``
    float in [0, 1]. Resizing and padding run on the GPU."""
    frames = frames_uint8.permute(0, 3, 1, 2).contiguous().to(dtype=dtype)
    _, _, h, w = frames.shape
    if (h, w) != (out_h, out_w):
        if mode in _INTERP_NEEDS_ALIGN:
            frames = F.interpolate(frames, size=(out_h, out_w), mode=mode, align_corners=False)
        else:
            frames = F.interpolate(frames, size=(out_h, out_w), mode=mode)
    frames = frames / 255.0
    if pad_h > 0 or pad_w > 0:
        frames = F.pad(frames, (0, pad_w, 0, pad_h), mode="constant", value=0)
    return frames.unsqueeze(0)


# --------------------------------------------------------------------------- #
# Output                                                                      #
# --------------------------------------------------------------------------- #

def crop_spatial_padding_ntchw(video, pad_h=0, pad_w=0):
    if video is None:
        return None
    if pad_h > 0:
        video = video[:, :, :, :-pad_h, :]
    if pad_w > 0:
        video = video[:, :, :, :, :-pad_w]
    return video


def ntchw_to_uint8_frames(video):
    if video is None or video.numel() == 0 or video.shape[1] == 0:
        return None
    video = video[0].permute(0, 2, 3, 1).contiguous()
    return (video * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()


def quality_to_crf(quality: int) -> int:
    q = max(0, min(100, int(quality)))
    return int(round((100 - q) * 51 / 100))


def open_stream_video_writer(output_path, fps=8, video_format="", preset="", quality=85):
    extra = ["-preset", str(preset)] if preset else []
    crf = quality_to_crf(quality)
    pix = "yuv444p" if video_format == "yuv444p" else "yuv420p"
    return imageio.get_writer(output_path, fps=fps, codec="libx265", pixelformat=pix,
                             macro_block_size=None, ffmpeg_params=["-crf", str(crf)] + extra)


def _normalize_png_name(name: str) -> str:
    name = str(name)
    return name if name.lower().endswith(".png") else f"{name}.png"


def append_chunk_to_png_dir(video_ntchw, output_dir, start_idx=0, pad_h=0, pad_w=0,
                            frame_names: Optional[List[str]] = None,
                            written_once: Optional[set] = None):
    os.makedirs(output_dir, exist_ok=True)
    frames = ntchw_to_uint8_frames(crop_spatial_padding_ntchw(video_ntchw, pad_h, pad_w))
    if frames is None:
        return 0, 0

    saved = 0
    for i, frame in enumerate(frames):
        idx = int(start_idx + i)
        if frame_names is not None:
            if idx >= len(frame_names):
                continue
            file_name = _normalize_png_name(frame_names[idx])
        else:
            file_name = f"{idx:05d}.png"
        out_path = os.path.join(output_dir, file_name)
        key = os.path.abspath(out_path)
        if written_once is not None and key in written_once:
            continue
        Image.fromarray(frame).save(out_path, format="PNG", compress_level=0, optimize=False)
        if written_once is not None:
            written_once.add(key)
        saved += 1
    return saved, int(frames.shape[0])
