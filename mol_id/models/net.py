import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import GatedMlp
from torch import nn

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.


@dataclass
class ModelArgs:
    dim: int = 320
    n_layers: int = 12
    n_heads: int = 20
    vocab_size: int = -1  # defined later by tokenizer
    dim_ffn: int = 960
    norm_eps: float = 1e-5
    multiple_of: int = 64
    padding_idx: int = -1  # defined later by tokenizer
    max_seq_len: int = 512
    initializer_range: float = 0.02

    def to_json(self):
        return self.__dict__

    def save_pretrained(self, output_dir: str):
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        (output_dir_path / "config.json").write_text(
            json.dumps(self.to_json(), indent=1)
        )

    @classmethod
    def from_pretrained(cls, output_dir: str):
        output_dir_path = Path(output_dir)
        config = json.loads((output_dir_path / "config.json").read_text())
        return cls(**config)


@torch.no_grad()
def init_weights_impl(module, initializer_range: float):
    """Initialize the weights"""
    if isinstance(module, nn.Linear):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            multiple_of (int): Multiple of the model dimension.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (LayerNorm): Layer normalization for attention output.
            ffn_norm (LayerNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.multiple_of = args.multiple_of
        self.attention = MHA(
            embed_dim=args.dim,
            num_heads=args.n_heads,
            qkv_proj_bias=False,
            out_proj_bias=False,
            dropout=0.0,
            causal=False,
            layer_idx=layer_id,
            use_flash_attn=True,
        )
        self.feed_forward = GatedMlp(
            in_features=args.dim,
            hidden_features=args.dim_ffn,
            out_features=args.dim,
            activation=F.silu,
            bias1=False,
            bias2=False,
            multiple_of=self.multiple_of,
        )
        self.layer_id = layer_id
        self.attention_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: int = None,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            cu_seqlens (torch.Tensor, optional): cumulated sequence lengths. Defaults to None.
            max_seqlen (int, optional): Maximum sequence length. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention.forward(
            self.attention_norm(x), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.register_buffer("encoding", torch.zeros(max_len, d_model))

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, cu_seqlens: torch.Tensor):
        """
        forward function of sinusoid encoding

        :param cu_seqlens: cumulative sequence lengths (batch_size+1, )
        """

        positions = torch.arange(cu_seqlens[-1])  # (total len,)
        offset = torch.repeat_interleave(
            cu_seqlens[:-1], cu_seqlens[1:] - cu_seqlens[:-1]
        )
        positions = positions - offset
        return self.encoding[positions, :]


@dataclass
class TransformerOutput:
    last_hidden_state: torch.Tensor
    logits: torch.Tensor | None = None


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            pos_embeddings (PositionalEncoding): Positional embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim, padding_idx=self.params.padding_idx
        )
        self.pos_embeddings = PositionalEncoding(params.dim, params.max_seq_len)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = nn.LayerNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: int = None,
        output_logits: bool = True,
    ):
        """
        Perform a forward pass through the Transformer model.

        Args:
            input_ids (torch.Tensor): Input tensor.
            cu_seqlens (torch.Tensor, optional): cumulated sequence lengths. Defaults to None.
            max_seqlen (int, optional): Maximum sequence length. Defaults to None.

        Returns:
            TransformerOutput: Output

        """
        h = self.tok_embeddings(input_ids)
        h += self.pos_embeddings(cu_seqlens)

        for layer in self.layers:
            h = layer(h, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        output = TransformerOutput(last_hidden_state=self.norm(h))
        if output_logits:
            output.logits = self.output(output.last_hidden_state)
        return output

    def save_pretrained(self, output_dir: str):
        self.params.save_pretrained(output_dir)
        torch.save(self.state_dict(), Path(output_dir) / "pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, output_dir: str):
        params = ModelArgs.from_pretrained(output_dir)
        model = cls(params)
        model.load_state_dict(
            torch.load(Path(output_dir) / "pytorch_model.bin", map_location="cpu")
        )
        return model
