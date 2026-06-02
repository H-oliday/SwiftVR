"""Causal chunk-wise streaming engine (chunk protocol + TAE/DiT wrappers)."""

from .chunk import ChunkType, ChunkSpec, build_chunk_specs
from .tae import StreamingTAE, apply_parallel_with_boundary
from .dit import StreamingDiT, INFERENCE_TIMESTEP, ROPE_EXTEND_MARGIN
