from functools import partial
from pathlib import Path

import typer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import BpeTrainer


def file_iterator(files: list[Path]):
    for file in files:
        with open(file, "r") as f:
            for line in f:
                yield line.split()[0]


def main(
    data_path: Path,
    regex_model: Path,
    save_path: Path,
    vocab_size: int = 2048,
):
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Split(regex_model.read_text(), "isolated")
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        special_tokens=["<unk>", "<cls>", "<sep>", "<pad>", "<mask>"],
    )
    files = list(data_path.glob("*.txt")) if data_path.is_dir() else [data_path]

    it = list(s for s in partial(file_iterator, files)())

    tokenizer.train_from_iterator(it, trainer, len(it))
    tokenizer.save(str(save_path))


if __name__ == "__main__":
    typer.run(main)
