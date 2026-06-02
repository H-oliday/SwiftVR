<h1 align="center">RVR: One-step Generative Streaming Real-time Video Restoration</h1>



<p align="center"><img src="assets/teaser.avif" width="100%" alt="RVR teaser"></p>



> **RVR** is the first generative video restoration model to reach **real-time 1080p streaming on a consumer-grade GPU** (≈26 FPS on a single RTX 5090), sustains **31 FPS at QHD (2560×1440)** and **14 FPS at 4K (3840×2160)** on a single H100, and streams at resolutions where every compared diffusion-based VR baseline runs out of memory.

<p>
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg?style=flat-square" alt="arXiv"></a>
  <a href="https://github.com/H-oliday/RVR/"><img src="https://img.shields.io/badge/Project-Page-1f8acb.svg?style=flat-square" alt="Project Page"></a>
  <a href="https://huggingface.co/H-oliday/RVR"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-ffce00.svg?style=flat-square" alt="HuggingFace"></a>
  <a href="https://github.com/H-oliday/RVR/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?style=flat-square" alt="License"></a>

</p>



## Updates
- [2026/06] Release the inference code and pretrained weights 🎉





## ✨ Highlights
- **Mask-free shifted-window self-attention (MFSWA).** Each spatial window is **pre-gathered into a dense tensor**, so every attention call reduces to a single standard scaled-dot-product (SDPA) call — *no attention mask, cyclic shift, or padding ever enters the graph*. This gives a **1.62× throughput gain over its full-attention teacher** at essentially identical quality, with **no dedicated sparse kernel**.
- **Restoration-aware Autoencoder (ReAE).** A lightweight encoder–decoder jointly fine-tuned with the DiT in pixel space removes the heavy-3D-VAE / tiled-decoding bottleneck.
- **Causal chunk-wise streaming.** A minimal causal protocol (no rolling KV cache, no overlapped DiT inference) bounds the temporal axis, confining the residual \(\mathcal{O}(N^2)\) cost to the spatial axes.
- **Kernel-agnostic & portable.** The same checkpoint runs **bit-identically** across PyTorch SDPA, FlashAttention-2/3, SageAttention, and xFormers — no retraining, weight conversion, or kernel rewrite.


## 📊 Results
### Efficiency at 2560×1440 (single H100, causal streaming, 24 frames)

| Metric | SeedVR2-3B (tile)| DOVE (tile)| FlashVSR-Tiny | **RVR (Ours)** |
|---|:---:|:---:|:---:|:---:|
| Avg. Time (s) ↓ | 17.320 | 27.615 | 2.493 | **0.766** |
| **FPS ↑** | 1.39 | 0.85 | 9.61 | **31.32** |
| Peak Mem. (GB) ↓ | 35.35 | 59.24 | 34.35 | 38.01 |

> At **3840×2160**, every compared diffusion-based VR baseline **OOMs** on a single H100; RVR sustains **14 FPS**.

### Qualitative comparison

<img src="assets/qualitative.png" width="100%" alt="RVR teaser">



## 🛠 Installation
```bash
git clone https://github.com/Holiday/RVR.git
cd RVR

conda create -n rvr python=3.10 -y
conda activate rvr

# Install PyTorch matching your CUDA toolkit first, e.g. CUDA 12.4:
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu124

# Install RVR (editable) and its dependencies:
pip install -e .
```

<details>
<summary><b>Hardware notes</b></summary>

- **Server:** single H100-80G reproduces the QHD/4K numbers above.
- **Consumer:** single RTX 5090 reaches ≈26 FPS at 1080p with the *same checkpoint* (default PyTorch SDPA path, bfloat16, causal chunk protocol).
- No hardware-specific retraining or kernel rewrite is required on any platform.
</details>



## 🗂 Model Zoo
| Model Name | Date | Backbone | Link |
|---|---|---|---|
| RVR | 2026.06 | Wan2.2-TI2V-5B | [🤗 HuggingFace](https://huggingface.co/H-oliday/RVR) |

```bash
huggingface-cli download H-oliday/RVR --local-dir checkpoints/
```

Expected checkpoint layout (the directory passed to `from_pretrained`):

```
checkpoints/
├── reae.safetensors            # Restoration-aware Autoencoder weights
├── prompt_embedding.safetensors# precomputed empty-prompt text embedding (key: "prompt_emb")
└── transformer/                # diffusers-format DiT
    ├── config.json
    └── diffusion_pytorch_model.safetensors
```


## 🚀 Quick Start
### Python API

```python
from rvr import RVRPipeline

pipe = RVRPipeline.from_pretrained("checkpoints/").to("cuda", dtype="bfloat16")

pipe.restore_video("low_quality.mp4", "restored.mp4", upscale=4)
```

`restore_video` also accepts an image folder as input and can write a PNG sequence
(`png_save=True`). Tunable knobs: `clip_len` (MIDDLE chunk size, multiple of 4),
`dit_overlap`, `fps`, `quality` (0–100, mapped to x265 CRF), `queue_size`.

### Streaming (causal, chunk by chunk, no future frames)

```python
session = pipe.stream(clip_len=24, resolution=(1920, 1080))

for lq_chunk in read_chunks("low_quality.mp4", n=24):   # lq_chunk: [T, H, W, 3] uint8
    hq = session.step(lq_chunk)        # [1, T', 3, H', W'] in [0, 1], or None if buffered
    if hq is not None:
        write(hq)

tail = session.flush()                 # flush the final buffered frames
```

### Command line

```bash
python scripts/inference.py \
  --input low_quality.mp4 \
  --output restored.mp4 \
  --checkpoint checkpoints/ \
  --upscale 4 \
  --clip-len 24 \
  --dtype bfloat16 \
```

Use `--png` to write a PNG sequence.




## 📁 Repository Structure
```
RVR/
├── README.md
├── requirements.txt
├── setup.py
├── scripts/
│   └── inference.py              # CLI entry point (thin wrapper over RVRPipeline)
└── rvr/
    ├── __init__.py               # exports RVRPipeline
    ├── pipeline.py               # RVRPipeline: from_pretrained / to / restore_video / stream
    ├── runner.py                 # four-stage pipelined runner (reader → H2D → GPU → writer)
    ├── io.py                     # frame reading, GPU preprocessing, mp4 / PNG writing
    ├── models/
    │   ├── reae.py               # ★ Restoration-aware Autoencoder
    │   └── transformer.py        # ★ DiT + mask-free shifted-window self-attention
    └── streaming/
        ├── chunk.py              # fixed-size causal chunk protocol
        ├── tae.py                # streaming autoencoder (causal boundary state)
        └── dit.py                # one-step streaming DiT (fixed timestep, RoPE offsets)
```

> ★ marks the two contribution-critical files: the MFSWA processor in `transformer.py` and `reae.py`.


## 📖 Citation
```bibtex
@article{yan2026rvr,
  title   = {RVR: One-step Generative Streaming Real-time Video Restoration},
  author  = {Yan, Jiaqi and Chen, Xiangyu and Zhong, Xinlin and Liu, Jie and Zhou, Jiantao and Li, Xuelong},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```



## 🙏 Acknowledgements
RVR builds on [Wan2.2-TI2V-5B](https://github.com/Wan-Video), the lightweight autoencoder [TAEHV](https://github.com/madebyollin/taehv), and the [RealBasicVSR](https://github.com/ckkelvinchan/RealBasicVSR) degradation pipeline. We thank the authors of [SeedVR2](https://github.com/ByteDance-Seed/SeedVR), [DOVE](https://github.com/zhengchen1999/DOVE), and [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) for releasing strong baselines, and the [UltraVideo](https://github.com/Tele-AI/UltraVideo) team for the training corpus.



## 📜 License
Released under the [Apache 2.0 License](LICENSE). The Wan2.2 backbone and any third-party weights remain subject to their original licenses.

<div align="center">
<sub>If RVR is useful to your research or product, please consider giving it a ⭐.</sub>
</div>
