from pathlib import Path

import pytorch_lightning as pl
import torch
import typer
import wandb
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from transformers import PreTrainedTokenizerFast

from .lightning_module.pretrain_module import PretrainLightningModule
from .models.net import ModelArgs
from .utils.helpers import get_time_str
from .utils.optim import AdamWConfig

PROJECT_NAME = "mol_id"

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def train(
    dataset_dir: str,
    tokenizer_file: str,
    # model config
    dim: int = 320,
    n_layers: int = 12,
    n_heads: int = 20,
    dim_ffn: int = 960,
    norm_eps: float = 0.00001,
    max_seq_len: int = 512,
    initializer_range: float = 0.02,
    # adamw config
    betas: tuple[float, float] = (0.9, 0.98),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    # lightning module config
    valid_size: float = 0.005,
    num_workers: int = 4,
    lr: float = 0.0004,
    min_lr: float = 0.00004,
    warmup_steps: int = 1000,
    num_cycles: float = 0.5,
    mask_prob: float = 0.15,
    mask_rand_prob: float = 0.1,
    mask_same_prob: float = 0.1,
    batch_size: int = 32,
    global_batch_size: int = 2048,
    epochs: int = 5,
    seed: int = 2486,
    # lightning trainer config
    gradient_clip_val: float = -1,
    ckpt_step: int = 4000,
    accelerator: str = "gpu",
    strategy: str = "auto",
    precision: str = "bf16-mixed",
    devices: int = 1,
    logdir: Path = Path("logs"),
    log_steps: int = 5,
    dev: bool = False,
    resume_ckpt: Path = None,
):
    assert (
        global_batch_size % (batch_size * devices) == 0
    ), "global_batch_size must be divisible by (batch_size * devices)"

    # special processing for gradient_clip_val
    if gradient_clip_val < 0.0:
        print(
            "**NOTE: gradient_clip_val is negative, setting it to None, make sure you know what you are doing!"
        )
        gradient_clip_val = None

    accumulator = global_batch_size // (batch_size * devices)
    pl.seed_everything(seed)
    version = get_time_str()
    logdir.mkdir(exist_ok=True, parents=True)
    wandb.login()

    print("version:", version)

    print("-- init config --")
    tok = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    model_config = ModelArgs(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        vocab_size=tok.vocab_size,
        dim_ffn=dim_ffn,
        norm_eps=norm_eps,
        padding_idx=tok.pad_token_id,
        max_seq_len=max_seq_len,
        initializer_range=initializer_range,
    )
    print("model_config", model_config)
    adamw_config = AdamWConfig(
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    print("adamw_config", adamw_config)

    print("-- init model --")
    model = PretrainLightningModule(
        config=model_config.__dict__,
        adamw_config=adamw_config.__dict__,
        tokenizer_file=tokenizer_file,
        dataset_dir=dataset_dir,
        valid_size=valid_size,
        num_workers=num_workers,
        lr=lr,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        num_cycles=num_cycles,
        mask_prob=mask_prob,
        mask_rand_prob=mask_rand_prob,
        mask_same_prob=mask_same_prob,
        batch_size=batch_size,
        global_batch_size=global_batch_size,
        epochs=epochs,
        seed=seed,
    )

    print("-- setup loggers --")
    wandb_logger = WandbLogger(
        name=f"{PROJECT_NAME}-pretrain-{version}",
        save_dir=logdir,
        project=f"{PROJECT_NAME}",
    )
    csv_logger = CSVLogger(
        save_dir=logdir,
        name=f"{PROJECT_NAME}_csv_logs",
        version=version,
    )
    logger_dir = Path(csv_logger.log_dir)

    print("-- setup callbacks --")
    epoch_checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=logger_dir / "epoch_ckpts",
        filename=PROJECT_NAME + "-pretrain-{epoch:02d}-{val_loss:.5f}",
        save_top_k=-1,
        save_last="link",
        mode="min",
    )
    step_checkpoint_callback = ModelCheckpoint(
        dirpath=logger_dir / "step_ckpts",
        filename=PROJECT_NAME + "-pretrain-{epoch:02d}-{step:06d}",
        save_top_k=-1,
        every_n_train_steps=ckpt_step,
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval="step")
    callbacks = [
        epoch_checkpoint_callback,
        step_checkpoint_callback,
        lr_monitor_callback,
    ]

    print("-- init trainer --")
    trainer = pl.Trainer(
        accelerator=accelerator,
        strategy=strategy,
        devices=devices,
        precision=precision,
        logger=[wandb_logger, csv_logger],
        callbacks=callbacks,
        max_epochs=epochs,
        accumulate_grad_batches=accumulator,
        gradient_clip_val=gradient_clip_val,
        log_every_n_steps=log_steps,
        fast_dev_run=dev,
    )
    trainer.fit(model, ckpt_path=resume_ckpt)


@app.command()
def placeholder():
    pass


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    app()
