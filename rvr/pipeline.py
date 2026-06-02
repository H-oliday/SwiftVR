"""High-level RVR pipeline.

    from rvr import RVRPipeline

    pipe = RVRPipeline.from_pretrained("checkpoints/").to("cuda", dtype="bfloat16")
    pipe.restore_video("low_quality.mp4", "restored.mp4", resolution=(1920, 1080))

The pipeline wraps the autoencoder, the one-step DiT and an empty text prompt
embedding, and exposes both an offline whole-file API (``restore_video``) and a
causal chunk-by-chunk API (``stream``).
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
from safetensors.torch import load_file

from .models import ReAE, WanTransformer3DModel
from .streaming import StreamingTAE, StreamingDiT
from .io import (
    get_video_info,
    selected_output_frame_names,
    preprocess_clip_uint8,
    crop_spatial_padding_ntchw,
    VIDEO_EXTS,
)
from .runner import run_pipeline, enable_max_fps_runtime


_DTYPES = {"float16": torch.float16, "fp16": torch.float16,
           "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
           "float32": torch.float32, "fp32": torch.float32}


def _as_dtype(dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower()
    if key not in _DTYPES:
        raise ValueError(f"Unsupported dtype {dtype!r}. Choose float16 / bfloat16 / float32.")
    return _DTYPES[key]


def _aligned_pad(size: int, multiple: int = 32) -> int:
    return (multiple - size % multiple) % multiple


class RVRPipeline:
    def __init__(self, reae, transformer, prompt_emb, upscale_mode: str = "bilinear"):
        self.reae = reae
        self.transformer = transformer
        self.prompt_emb = prompt_emb
        self.upscale_mode = upscale_mode

        self.tae_stream = StreamingTAE(reae)
        self.dit_stream = StreamingDiT(transformer, overlap=0)

        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self._prepared = False

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir,
        *,
        reae_filename: str = "reae.safetensors",
        transformer_subfolder: str = "transformer",
        prompt_embedding_filename: str = "prompt_embedding.safetensors",
        upscale_mode: str = "bilinear",
        device=None,
        dtype=None,
    ) -> "RVRPipeline":
        root = Path(checkpoint_dir)

        reae = ReAE(str(root / reae_filename))
        transformer = WanTransformer3DModel.from_pretrained(str(root), subfolder=transformer_subfolder)
        prompt_emb = load_file(str(root / prompt_embedding_filename))["prompt_emb"][0]

        pipe = cls(reae, transformer, prompt_emb, upscale_mode=upscale_mode)
        if device is not None or dtype is not None:
            pipe.to(device or "cpu", dtype=dtype or "float32")
        return pipe

    def to(self, device=None, dtype=None, *, attention_backend="auto", torch_compile=False):
        """Move the models to ``device``/``dtype`` and prepare them for inference
        (fused projections + shifted-window self-attention, once)."""
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = _as_dtype(dtype)

        self.reae.to(self.device, self.dtype).eval()
        self.transformer.to(self.device, self.dtype).eval()

        enable_max_fps_runtime(allow_tf32=True)
        if not self._prepared and hasattr(self.transformer, "prepare_for_inference"):
            self.transformer.prepare_for_inference(
                attention_backend=attention_backend,
                use_torch_compile=torch_compile,
                compile_mode="default")
            self._prepared = True
        return self

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _target_size(self, lq_h, lq_w, resolution, upscale):
        if resolution is not None:
            out_w, out_h = int(resolution[0]), int(resolution[1])
        else:
            out_h, out_w = lq_h * upscale, lq_w * upscale
        return out_h, out_w, _aligned_pad(out_h), _aligned_pad(out_w)

    @staticmethod
    def _resolve_output(input_path: Path, output_path: Path, png_save: bool):
        if png_save:
            output_path.mkdir(parents=True, exist_ok=True)
            return output_path / f"{input_path.stem}.mp4", output_path
        if output_path.suffix.lower() in VIDEO_EXTS:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return output_path, output_path.parent
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / f"{input_path.stem}.mp4", output_path

    # ------------------------------------------------------------------ #
    # Offline (whole file)                                               #
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def restore_video(
        self,
        input_path,
        output_path,
        *,
        resolution: Optional[Tuple[int, int]] = None,
        upscale: int = 4,
        clip_len: int = 24,
        dit_overlap: int = 0,
        fps: Optional[float] = None,
        quality: int = 85,
        png_save: bool = False,
        save_format: str = "",
        ffmpeg_preset: str = "",
        queue_size: int = 3,
        verbose: bool = True,
    ) -> dict:
        """Restore a whole video file or image folder.

        ``resolution`` is the output ``(width, height)``; if omitted the output
        is the low-quality input upscaled by ``upscale``. ``clip_len`` must be a
        multiple of 4.
        """
        if clip_len % 4 != 0:
            raise ValueError(f"clip_len must be a multiple of 4, got {clip_len}")
        if not self._prepared:
            self.to()

        input_path = Path(input_path)
        output_path = Path(output_path)

        raw_total, lq_h, lq_w, src_fps = get_video_info(input_path, fallback_fps=fps or 30)
        total_frames = 4 * ((raw_total - 1) // 4) + 1

        out_h, out_w, pad_h, pad_w = self._target_size(lq_h, lq_w, resolution, upscale)
        final_video_path, png_output_dir = self._resolve_output(input_path, output_path, png_save)
        png_frame_names = (selected_output_frame_names(input_path)
                           if (png_save and input_path.is_dir()) else None)

        self.dit_stream.overlap = dit_overlap

        written, wall = run_pipeline(
            video_path=input_path,
            final_output_path=str(final_video_path),
            png_output_dir=str(png_output_dir),
            tae_stream=self.tae_stream,
            dit_stream=self.dit_stream,
            prompt_emb=self.prompt_emb,
            device=self.device,
            dtype=self.dtype,
            total_frames=total_frames,
            clip_len=clip_len,
            lq_h=lq_h, lq_w=lq_w,
            out_h=out_h, out_w=out_w, pad_h=pad_h, pad_w=pad_w,
            upscale_mode=self.upscale_mode,
            source_fps=(fps or src_fps),
            png_save=png_save,
            quality=quality,
            save_format=save_format,
            ffmpeg_preset=ffmpeg_preset,
            queue_size=queue_size,
            png_frame_names=png_frame_names,
            verbose=verbose,
        )
        return {"frames": written, "seconds": wall,
                "fps": (written / wall if wall > 0 else 0.0),
                "output": str(png_output_dir if png_save else final_video_path)}

    # ------------------------------------------------------------------ #
    # Streaming (chunk by chunk, causal)                                 #
    # ------------------------------------------------------------------ #

    def stream(self, *, clip_len: int = 24, resolution: Optional[Tuple[int, int]] = None,
               upscale: int = 4, dit_overlap: int = 1) -> "StreamSession":
        if clip_len % 4 != 0:
            raise ValueError(f"clip_len must be a multiple of 4, got {clip_len}")
        if not self._prepared:
            self.to()
        return StreamSession(self, clip_len=clip_len, resolution=resolution,
                             upscale=upscale, dit_overlap=dit_overlap)


class StreamSession:
    """Causal chunk-by-chunk session. Call ``step`` with each new clip of frames
    and ``flush`` once at the end. Output sizing is taken from ``resolution`` or
    inferred from the first clip via ``upscale``."""

    def __init__(self, pipe: RVRPipeline, clip_len, resolution, upscale, dit_overlap):
        self.pipe = pipe
        self.clip_len = clip_len
        self.resolution = resolution
        self.upscale = upscale
        self._sizes = None

        pipe.tae_stream.reset()
        pipe.dit_stream.reset()
        pipe.dit_stream.overlap = dit_overlap

    def _ensure_sizes(self, lq_h, lq_w):
        if self._sizes is None:
            self._sizes = self.pipe._target_size(lq_h, lq_w, self.resolution, self.upscale)
        return self._sizes

    def _run_latents(self, z):
        z_bcfhw = z.permute(0, 2, 1, 3, 4).contiguous()
        den = self.pipe.dit_stream.denoise(z_bcfhw, self.pipe.prompt_emb)
        return den.permute(0, 2, 1, 3, 4).contiguous()

    @torch.inference_mode()
    def step(self, frames_uint8: torch.Tensor) -> Optional[torch.Tensor]:
        """``frames_uint8``: ``[T, H, W, 3]`` uint8. Returns ``[1, T', 3, H', W']``
        in [0, 1], or ``None`` if the frames were buffered (T not a multiple of 4)."""
        g = frames_uint8.to(self.pipe.device)
        out_h, out_w, pad_h, pad_w = self._ensure_sizes(g.shape[1], g.shape[2])
        clip = preprocess_clip_uint8(g, out_h, out_w, self.pipe.upscale_mode, pad_h, pad_w, self.pipe.dtype)

        z = self.pipe.tae_stream.encode_chunk(clip)
        if z is None:
            return None
        z_ntchw = self._run_latents(z)
        rgb = self.pipe.tae_stream.decode_chunk(z_ntchw)
        return crop_spatial_padding_ntchw(rgb, pad_h, pad_w)

    @torch.inference_mode()
    def flush(self) -> Optional[torch.Tensor]:
        z = self.pipe.tae_stream.flush_encoder()
        if z is None or self._sizes is None:
            return None
        _, _, pad_h, pad_w = self._sizes
        z_ntchw = self._run_latents(z)
        rgb = self.pipe.tae_stream.decode_chunk(z_ntchw)
        return crop_spatial_padding_ntchw(rgb, pad_h, pad_w)
