import dataclasses

import torch
from flash_attn.modules.mha import MHA


@dataclasses.dataclass
class TransformerConfig:
    d_model: int
    n_head: int
    n_layers: int
    d_ffn: int
