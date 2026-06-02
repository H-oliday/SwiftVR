"""Model definitions for RVR: the DiT backbone and the autoencoder."""

from .reae import ReAE, MemBlock, TPool, TGrow, Clamp
from .transformer import (
    WanTransformer3DModel,
    enable_shifted_window_self_attention,
    compile_transformer_blocks,
    set_attention_backend,
    get_attention_backend,
    list_available_attention_backends,
)
