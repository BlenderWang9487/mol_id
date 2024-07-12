from dataclasses import dataclass
from datetime import datetime
from functools import partial

import numpy as np
import torch
from torch.nn import functional as F
from transformers import PreTrainedTokenizerFast


def get_time_str():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


@dataclass
class CollateOutput:
    input_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    special_tokens_mask: torch.Tensor


class SmilesCollator:

    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_len: int):
        self.tokenizer = tokenizer
        self.mask_id = tokenizer.convert_tokens_to_ids("<mask>")
        self.max_len = max_len

    def collate_impl(
        self,
        batch: list[str],
    ):
        """
        Collates a batch of single cell expression data to be used in the model.

        Args:
            batch: A list of dictionaries representing the batch of data.

        Returns:
            A dictionary containing the collated data:
            - "input_ids": collated gene input IDs.
            - "cu_seqlens": cumulative sequence lengths (for flash attn).
            - "max_seqlen": maximum sequence length.
        """
        tok_output = self.tokenizer.batch_encode_plus(
            batch,
            add_special_tokens=True,
            return_length=True,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_special_tokens_mask=True,
            max_length=self.max_len,
            truncation=True,
        )
        lengths = tok_output["length"]

        # for flash attention varlen interface
        cu_seqlens = F.pad(
            torch.tensor(lengths, dtype=torch.int32).cumsum(0, dtype=torch.int32),
            (1, 0),  # pad 0 before the first element
        )
        max_seqlen = int(np.max(lengths))

        cu_input_ids = torch.from_numpy(
            np.concatenate(tok_output["input_ids"]),
        )
        cu_special_tokens_mask = torch.from_numpy(
            np.concatenate(tok_output["special_tokens_mask"]),
        )

        return CollateOutput(
            input_ids=cu_input_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            special_tokens_mask=cu_special_tokens_mask,
        )

    def __call__(self, batch: list[dict]):
        texts = [x["smiles"] for x in batch]
        return self.collate_impl(texts)


def make_mlm_input_impl(
    input_ids: torch.LongTensor,
    mask_prob: float,
    mask_rand_prob: float,
    mask_same_prob: float,
    rand_low: int,
    rand_high: int,
    special_tokens_mask: torch.BoolTensor,
    mask_idx: int,
) -> tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]:
    """
    Mostly referenced from transformers repo:
    https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L634

    Args:
        input_ids (torch.LongTensor)
        mask_prob (float): total mask probability
        mask_rand_prob (float): on the masked token, the probability to replace with a random token
        mask_same_prob (float): on the masked token, the probability to keep the same token
        rand_low (int): random token low (inclusive)
        rand_high (int): random token high (exclusive)
        special_tokens_mask (torch.BoolTensor): mask for special tokens (1 for special tokens, 0 for normal tokens)
        mask_idx (int): mask token index

    Returns:
        tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]: input_ids, labels, masked_indices
            input_ids: (total_lens,) masked input ids
            labels: (total_lens,) labels for MLM
            masked_indices: (total_lens,) True if the token is masked
    """
    labels = input_ids.clone()
    device = input_ids.device

    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mask_prob, device=device)

    probability_matrix.masked_fill_(
        special_tokens_mask, value=0.0
    )  # no mask for special tokens

    masked_indices = torch.bernoulli(probability_matrix).bool()

    # (1 - mask_rand_prob - mask_same_prob)% of the time, we replace masked input tokens with mask_token <mask>
    replaced_prob = 1.0 - mask_rand_prob - mask_same_prob
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, replaced_prob, device=device)).bool()
        & masked_indices
    )
    input_ids[indices_replaced] = mask_idx

    # (mask_rand_prob)% of the time, we replace masked input tokens with random word
    random_prob = mask_rand_prob / (mask_rand_prob + mask_same_prob)
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, random_prob, device=device)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(
        low=rand_low,
        high=rand_high,
        size=labels.shape,
        dtype=torch.long,
        device=device,
    )
    input_ids[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return input_ids, labels, masked_indices


def check_colab():
    """
    Check if the code is running in Google Colab.

    Returns:
        _type_: _description_
    """
    try:
        from google.colab import userdata

        is_colab = True
    except ImportError:
        is_colab = False
    return is_colab
