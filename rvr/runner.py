"""Four-stage pipelined runner: reader -> H2D -> GPU -> writer.

Overlaps host reading, host->device copy, GPU restoration and disk writing on
separate threads/CUDA streams to maximise sustained throughput.
"""

import time
import queue
import threading
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from .streaming.chunk import ChunkType
from .io import (
    iter_video_clips_fixed_scheme,
    preprocess_clip_uint8,
    crop_spatial_padding_ntchw,
    ntchw_to_uint8_frames,
    append_chunk_to_png_dir,
    open_stream_video_writer,
)


def cuda_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def enable_max_fps_runtime(allow_tf32=True):
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _event_elapsed_seconds(start_event, end_event):
    try:
        end_event.synchronize()
        return start_event.elapsed_time(end_event) / 1000.0
    except Exception:
        return 0.0


@dataclass
class _Item:
    clip_idx: int
    spec: object = None
    cpu_rgb: Optional[torch.Tensor] = None
    gpu_rgb: Optional[torch.Tensor] = None
    rgb_out_gpu: Optional[torch.Tensor] = None
    rgb_out_cpu: Optional[torch.Tensor] = None
    h2d_event: Optional[torch.cuda.Event] = None
    d2h_event: Optional[torch.cuda.Event] = None
    timings: Dict[str, float] = field(default_factory=dict)
    stop: bool = False


def _stop_item():
    return _Item(clip_idx=-1, stop=True)


def run_pipeline(
    *,
    video_path,
    final_output_path,
    png_output_dir,
    tae_stream,
    dit_stream,
    prompt_emb,
    device,
    dtype,
    total_frames: int,
    clip_len: int,
    lq_h: int,
    lq_w: int,
    out_h: int,
    out_w: int,
    pad_h: int,
    pad_w: int,
    upscale_mode: str,
    source_fps: float,
    png_save: bool,
    quality: int,
    save_format: str = "",
    ffmpeg_preset: str = "",
    queue_size: int = 3,
    png_frame_names: Optional[List[str]] = None,
    verbose: bool = True,
):
    q_read = queue.Queue(maxsize=max(1, queue_size))
    q_gpu = queue.Queue(maxsize=max(1, queue_size))
    q_write = queue.Queue(maxsize=max(1, queue_size))

    stage_errors = []
    stop_event = threading.Event()
    frames_state = {"next_idx": 0, "saved": 0}
    png_written_once = set()

    use_cuda = torch.cuda.is_available() and device.type == "cuda"
    h2d_stream = torch.cuda.Stream(device=device) if use_cuda else None
    d2h_stream = torch.cuda.Stream(device=device) if use_cuda else None

    def record_error(stage_name):
        stage_errors.append((stage_name, traceback.format_exc()))
        stop_event.set()
        for q in (q_read, q_gpu, q_write):
            try:
                q.put(_stop_item())
            except Exception:
                pass

    def reader_worker():
        try:
            clips = iter_video_clips_fixed_scheme(
                video_path, clip_len=clip_len, total_frames=total_frames, crop_h=lq_h, crop_w=lq_w)
            for spec, cpu_rgb in clips:
                if stop_event.is_set():
                    break
                try:
                    cpu_rgb = cpu_rgb.pin_memory()
                except Exception:
                    pass
                q_read.put(_Item(clip_idx=spec.clip_idx, spec=spec, cpu_rgb=cpu_rgb))
            q_read.put(_stop_item())
        except Exception:
            record_error("reader")

    def h2d_worker():
        try:
            while True:
                item = q_read.get()
                if item.stop:
                    q_gpu.put(_stop_item())
                    break
                if stop_event.is_set():
                    continue
                if use_cuda:
                    se = torch.cuda.Event(enable_timing=True)
                    ee = torch.cuda.Event(enable_timing=True)
                    with torch.cuda.stream(h2d_stream):
                        se.record(h2d_stream)
                        item.gpu_rgb = item.cpu_rgb.to(device=device, non_blocking=True)
                        ee.record(h2d_stream)
                    item.h2d_event = ee
                    item.timings["_h2d"] = (se, ee)
                else:
                    item.gpu_rgb = item.cpu_rgb.to(device=device)
                item.cpu_rgb = None
                q_gpu.put(item)
        except Exception:
            record_error("h2d")

    def _start_d2h(item):
        if item.rgb_out_gpu is None or item.rgb_out_gpu.shape[1] == 0:
            item.rgb_out_cpu = None
            return item
        try:
            cpu_buf = torch.empty(item.rgb_out_gpu.shape, dtype=item.rgb_out_gpu.dtype,
                                  device="cpu", pin_memory=True)
        except Exception:
            cpu_buf = torch.empty(item.rgb_out_gpu.shape, dtype=item.rgb_out_gpu.dtype, device="cpu")
        if use_cuda:
            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(d2h_stream):
                se.record(d2h_stream)
                cpu_buf.copy_(item.rgb_out_gpu, non_blocking=True)
                try:
                    item.rgb_out_gpu.record_stream(d2h_stream)
                except Exception:
                    pass
                ee.record(d2h_stream)
            item.d2h_event = ee
        else:
            cpu_buf.copy_(item.rgb_out_gpu)
        item.rgb_out_cpu = cpu_buf
        return item

    def gpu_worker():
        try:
            tae_stream.reset()
            dit_stream.reset()
            n_lat = clip_len // 4
            prev_dit_out_cpu = None

            while True:
                item = q_gpu.get()
                if item.stop:
                    break
                if stop_event.is_set():
                    continue

                if item.h2d_event is not None:
                    torch.cuda.current_stream(device=device).wait_event(item.h2d_event)
                spec = item.spec

                if use_cuda:
                    t_start = torch.cuda.Event(enable_timing=True)
                    t_end = torch.cuda.Event(enable_timing=True)
                    t_start.record(torch.cuda.current_stream(device=device))
                else:
                    t0 = time.perf_counter()

                clip_rgb = preprocess_clip_uint8(
                    item.gpu_rgb, out_h=out_h, out_w=out_w, mode=upscale_mode,
                    pad_h=pad_h, pad_w=pad_w, dtype=dtype)
                z = tae_stream.encode_chunk_fixed(clip_rgb, spec)

                if spec.ctype == ChunkType.LAST:
                    z_ntchw = dit_stream.denoise_last_chunk(
                        z, spec, prompt_emb, prev_dit_out_cpu, n_lat, device, dtype)
                else:
                    z_bcfhw = z.permute(0, 2, 1, 3, 4).contiguous()
                    z_den = dit_stream.denoise(z_bcfhw, prompt_emb)
                    z_ntchw = z_den.permute(0, 2, 1, 3, 4).contiguous()
                    prev_dit_out_cpu = z_bcfhw[:, :, -n_lat:].detach().cpu().clone()

                rgb_out = tae_stream.decode_chunk_fixed(z_ntchw, spec)
                if rgb_out is not None and rgb_out.shape[1] > 0:
                    item.rgb_out_gpu = crop_spatial_padding_ntchw(rgb_out, pad_h, pad_w).detach()
                else:
                    item.rgb_out_gpu = None

                if use_cuda:
                    t_end.record(torch.cuda.current_stream(device=device))
                    item.timings["gpu"] = _event_elapsed_seconds(t_start, t_end)
                else:
                    item.timings["gpu"] = time.perf_counter() - t0

                if verbose:
                    out_n = 0 if item.rgb_out_gpu is None else item.rgb_out_gpu.shape[1]
                    fps = out_n / item.timings["gpu"] if item.timings["gpu"] > 0 else 0.0
                    print(f"  [gpu] {spec.ctype.value:6s} clip {item.clip_idx}: "
                          f"out={out_n}f time={item.timings['gpu']:.3f}s gpu_fps={fps:.2f}")

                item.gpu_rgb = None
                q_write.put(_start_d2h(item))
                del clip_rgb, z, z_ntchw, rgb_out

            q_write.put(_stop_item())
        except Exception:
            record_error("gpu")

    def writer_worker():
        writer = None
        try:
            while True:
                item = q_write.get()
                if item.stop:
                    break
                if stop_event.is_set():
                    continue
                if item.d2h_event is not None:
                    item.d2h_event.synchronize()
                item.rgb_out_gpu = None

                n_written = n_consumed = 0
                if item.rgb_out_cpu is not None and item.rgb_out_cpu.shape[1] > 0:
                    if png_save:
                        n_written, n_consumed = append_chunk_to_png_dir(
                            item.rgb_out_cpu, png_output_dir, start_idx=frames_state["next_idx"],
                            frame_names=png_frame_names, written_once=png_written_once)
                    else:
                        if writer is None:
                            writer = open_stream_video_writer(
                                final_output_path, fps=source_fps, video_format=save_format,
                                preset=ffmpeg_preset, quality=quality)
                        frames = ntchw_to_uint8_frames(item.rgb_out_cpu)
                        if frames is not None:
                            for frame in frames:
                                writer.append_data(frame)
                            n_written = n_consumed = int(frames.shape[0])

                frames_state["next_idx"] += n_consumed
                frames_state["saved"] += n_written
        except Exception:
            record_error("writer")
        finally:
            if writer is not None:
                writer.close()

    threads = [
        threading.Thread(target=reader_worker, name="reader", daemon=True),
        threading.Thread(target=h2d_worker, name="h2d", daemon=True),
        threading.Thread(target=gpu_worker, name="gpu", daemon=True),
        threading.Thread(target=writer_worker, name="writer", daemon=True),
    ]
    t0 = time.perf_counter()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall_time = time.perf_counter() - t0

    if stage_errors:
        name, err = stage_errors[0]
        raise RuntimeError(f"Pipeline stage '{name}' failed:\n{err}")

    written = frames_state["saved"] if png_save else frames_state["next_idx"]
    return written, wall_time
