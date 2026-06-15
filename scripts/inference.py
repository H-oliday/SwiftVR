"""Command-line entry point for SwiftVR inference.

Thin wrapper around ``swiftvr.SwiftVRPipeline``; all defaults live in
``SwiftVRPipeline.restore_video``.

    python script/inference.py \
        --input low_quality.mp4 --output restored.mp4 \
        --checkpoint checkpoints/ --upscale 4 --clip-len 24 --dtype bfloat16
"""

import argparse

from swiftvr import SwiftVRPipeline


def _parse_resolution(value):
    if value is None:
        return None
    w, h = value.lower().split("x")
    return int(w), int(h)


def build_parser():
    p = argparse.ArgumentParser(description="SwiftVR streaming video restoration.")

    p.add_argument("--input", required=True, help="Low-quality video file or image folder.")
    p.add_argument("--output", required=True, help="Output mp4 path or directory.")
    p.add_argument("--checkpoint", required=True, help="Checkpoint directory (see README layout).")

    p.add_argument("--resolution", type=str, default=None,
                   help="Output resolution as WxH (e.g. 1920x1080). Overrides --upscale.")
    p.add_argument("--upscale", type=int, default=4, help="Upscale factor when --resolution is unset.")
    p.add_argument("--clip-len", type=int, default=24, help="MIDDLE chunk size (multiple of 4).")
    p.add_argument("--dit-overlap", type=int, default=0, help="Temporal overlap (latents) for blending.")

    p.add_argument("--fps", type=float, default=None, help="Output fps (defaults to source fps).")
    p.add_argument("--quality", type=int, default=60, help="Output quality 0-100 (maps to x265 CRF).")
    p.add_argument("--png", action="store_true", help="Write a PNG sequence instead of an mp4.")
    p.add_argument("--save-format", type=str, default="", help="Set to 'yuv444p' for 4:4:4 mp4.")
    p.add_argument("--ffmpeg-preset", type=str, default="", help="x265 preset (e.g. fast, medium).")
    p.add_argument("--queue-size", type=int, default=3, help="Pipeline queue depth.")
    p.add_argument("--attention_backend", type=str, default="auto", choices=["auto", "sdpa", "flash_attn_2", "flash_attn_3", "sageattention", "xformers"],
                    help="Attention backend. 'auto' lets SwiftVR pick the fastest available backend.",)
    p.add_argument("--torch_compile", action="store_true", help="Enable torch.compile. Disabled by default to avoid long recompilation on dynamic paths.",)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--quiet", action="store_true")
    return p


def main():
    args = build_parser().parse_args()

    pipe = SwiftVRPipeline.from_pretrained(args.checkpoint).to(
        args.device, dtype=args.dtype,
        attention_backend=args.attention_backend,
        torch_compile=args.torch_compile)

    stats = pipe.restore_video(
        args.input, args.output,
        resolution=_parse_resolution(args.resolution),
        upscale=args.upscale,
        clip_len=args.clip_len,
        dit_overlap=args.dit_overlap,
        fps=args.fps,
        quality=args.quality,
        png_save=args.png,
        save_format=args.save_format,
        ffmpeg_preset=args.ffmpeg_preset,
        queue_size=args.queue_size,
        verbose=not args.quiet,
    )

    print(f"\nDone. {stats['frames']} frames in {stats['seconds']:.2f}s "
          f"({stats['fps']:.2f} fps) -> {stats['output']}")


if __name__ == "__main__":
    main()
