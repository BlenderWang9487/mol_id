from functools import partial

import datasets
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from ..models.net import (ModelArgs, Transformer, TransformerOutput,
                          init_weights_impl)
from ..utils.helpers import SmilesCollator, make_mlm_input_impl
from ..utils.optim import (AdamWConfig,
                           get_cosine_schedule_with_warmup_and_min_lr_lambda)


class PretrainLightningModule(pl.LightningModule):
    def __init__(
        self,
        # config
        config: dict,
        adamw_config: dict,
        # tokenizer
        tokenizer_file: str,
        # dataset
        dataset_dir: str,
        valid_size: float = 0.005,
        num_workers: int = 4,
        # training
        lr: float = 4e-4,
        min_lr: float = 4e-5,
        warmup_steps: int = 1000,
        num_cycles: float = 0.5,
        mask_prob: float = 0.15,
        mask_rand_prob: float = 0.1,
        mask_same_prob: float = 0.1,
        batch_size: int = 32,
        global_batch_size: int = 2048,
        epochs: int = 5,
        # reproducibility
        seed: int = 2486,
    ):
        super().__init__()
        assert 0 <= mask_prob <= 1, "mask_prob should be in [0, 1]"
        assert 0 <= mask_rand_prob <= 1, "mask_rand_prob should be in [0, 1]"
        assert 0 <= mask_same_prob <= 1, "mask_same_prob should be in [0, 1]"
        assert (
            mask_rand_prob + mask_same_prob <= 1
        ), "mask_rand_prob + mask_same_prob should be less than 1"
        self.save_hyperparameters()

        self.model_config = ModelArgs(**config)
        self.model = Transformer(self.model_config)
        self.loss_fn = F.cross_entropy

        self.adamw_config = AdamWConfig(**adamw_config)

        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
        self.collator = SmilesCollator(
            self.tokenizer, max_len=self.model_config.max_seq_len
        )

        mask_token_id = self.tokenizer.convert_tokens_to_ids('<mask>')
        self.mlm_input_maker = partial(
            make_mlm_input_impl,
            mask_prob=mask_prob,
            mask_rand_prob=mask_rand_prob,
            mask_same_prob=mask_same_prob,
            rand_low=mask_token_id + 1,
            rand_high=self.tokenizer.vocab_size,
            mask_idx=mask_token_id,
        )

        # init weight
        self.model.apply(
            partial(
                init_weights_impl, initializer_range=self.model_config.initializer_range
            )
        )

    @property
    def my_dataset(self) -> datasets.Dataset:
        if not hasattr(self, "_my_dataset"):
            self._my_dataset = datasets.load_from_disk(self.hparams.dataset_dir)
        return self._my_dataset

    def setup(self, stage):
        if stage == "fit":
            training_set = self.my_dataset.train_test_split(
                test_size=self.hparams.valid_size, seed=self.hparams.seed
            )
            self.train_set, self.val_set = training_set["train"], training_set["test"]

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
        )

    def _make_mlm_input(
        self,
        input_ids: torch.LongTensor,
        special_tokens_mask: torch.BoolTensor,
    ):
        input_ids, labels, masked_indices = self.mlm_input_maker(
            input_ids, special_tokens_mask
        )
        labels = labels[masked_indices]  # (total_lens,) -> (masked nnz, )
        return input_ids, labels, masked_indices

    def forward(
        self,
        input_ids: torch.LongTensor,
        special_tokens_mask: torch.BoolTensor,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: int = None,
        **kwargs,
    ):
        input_ids, labels, masked_tokens_mask = self._make_mlm_input(
            input_ids, special_tokens_mask
        )
        model_output = self.model.forward(
            input_ids=input_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            output_logits=False,
        )
        logits = self.model.output.forward(
            model_output.last_hidden_state[
                masked_tokens_mask
            ]  # only calculate logits of masked tokens to save computation
        )
        loss = self.loss_fn(logits, labels)
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self.forward(**batch)
        batch_size = batch["cu_seqlens"].shape[0] - 1
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        # report acc
        pred = logits.argmax(dim=-1)
        acc = (pred == labels).float().mean()
        self.log(
            "train_acc",
            acc,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        loss, logits, labels = self.forward(**batch)
        batch_size = batch["cu_seqlens"].shape[0] - 1
        self.log(
            "val_loss",
            loss,
            sync_dist=True,
            batch_size=batch_size,
        )
        # report acc
        pred = logits.argmax(dim=-1)
        acc = (pred == labels).float().mean()
        self.log(
            "val_acc",
            acc,
            sync_dist=True,
            batch_size=batch_size,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.adamw_config.weight_decay,
            betas=self.adamw_config.betas,
            eps=self.adamw_config.eps,
        )

        num_training_steps = (
            len(self.my_dataset) // self.hparams.global_batch_size * self.hparams.epochs
        )

        lr_lambda = partial(
            get_cosine_schedule_with_warmup_and_min_lr_lambda,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=self.hparams.num_cycles,
            lr=self.hparams.lr,
            min_lr=self.hparams.min_lr,
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, -1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
