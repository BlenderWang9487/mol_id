NOTE = """
DeprecationWarning:

We are using molformer tokenizer now. This script is for reference only.

We notice that using BPE to train a tokenizer will results
"""

from functools import partial
from pathlib import Path

import typer
from tokenizers import Encoding, Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import RobertaProcessing
from tokenizers.trainers import BpeTrainer


def file_iterator(files: list[Path]):
    for file in files:
        with open(file, "r") as f:
            for line in f:
                yield line.split()[0]


app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def train(
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


@app.command()
def add_postprocessor(
    tokenizer_file: Path,
    save_path: Path,
):

    test_smi = "c1cc2c(nc1)n(c(=O)o2)CC[NH+]3CCCCC3"
    print("test smiles:", test_smi)

    def test_tok(tokenizer: Tokenizer, test_smi: str):
        tok_output: Encoding = tokenizer.encode(test_smi)
        print("encoded output:", tok_output.ids)

        decoded_output = tokenizer.decode(tok_output.ids, skip_special_tokens=False)
        print("decoded output:", decoded_output)

    tokenizer: Tokenizer = Tokenizer.from_file(str(tokenizer_file))
    print("before adding postprocessor:")
    test_tok(tokenizer, test_smi)

    sep_str_and_id = ("<sep>", tokenizer.token_to_id("<sep>"))
    cls_str_and_id = ("<cls>", tokenizer.token_to_id("<cls>"))

    tokenizer.post_processor = RobertaProcessing(
        sep=sep_str_and_id, cls=cls_str_and_id, add_prefix_space=False
    )
    print("after adding postprocessor:")
    test_tok(tokenizer, test_smi)
    tokenizer.save(str(save_path))


@app.command()
def to_huggingface(
    tokenizer_file: Path,
    save_path: Path,
):
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_file),
    )

    # explicitly set special tokens
    tokenizer.unk_token = "<unk>"
    tokenizer.cls_token = "<cls>"
    tokenizer.sep_token = "<sep>"
    tokenizer.pad_token = "<pad>"
    tokenizer.mask_token = "<mask>"
    tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    from warnings import warn

    warn(NOTE)
    app()
