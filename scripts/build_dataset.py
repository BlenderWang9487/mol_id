from pathlib import Path

import datasets
import numpy as np
import typer
from rdkit import Chem
from transformers import PreTrainedTokenizerFast


def file_iterator(files: list[Path]):
    for file in files:
        with open(file, "r") as f:
            for line in f:
                smiles = line.split()[0]
                yield {"smiles": smiles, "strlen": len(smiles)}


app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def build(
    data_path: Path,
    save_path: Path,
    num_proc: int = 4,
    num_shards: int = 10,
):
    files = list(data_path.glob("*.txt")) if data_path.is_dir() else [data_path]

    dataset = datasets.Dataset.from_generator(
        file_iterator,
        gen_kwargs={
            "files": files,
        },
        num_proc=num_proc,
    )

    dataset.set_format("np")
    dataset.save_to_disk(save_path, num_shards=num_shards, num_proc=num_proc)


@app.command()
def filter(
    dataset_path: Path,
    save_path: Path,
    pretrained_tokenizer: Path,
    max_len: int = 512,  # after tokenization
    min_len: int = 8,  # after tokenization
    filter_unk: bool = True,
    num_proc: int = 4,
    num_shards: int = 10,
):
    tok: PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained(
        pretrained_tokenizer
    )
    unk_id = tok.unk_token_id
    print("### unk_id:", unk_id)

    dataset = datasets.load_from_disk(dataset_path)
    dataset.set_format("np")
    print("before filter:", dataset)

    def filter_fn(smiles_list):
        tok_output = tok.batch_encode_plus(
            smiles_list.tolist(),
            return_token_type_ids=False,
            return_attention_mask=False,
            return_length=True,
        )
        lens = np.array(tok_output["length"])
        len_filter = (lens <= max_len) & (lens >= min_len)
        if filter_unk:
            unk_filter = np.array([not (unk_id in x) for x in tok_output["input_ids"]])
            return len_filter & unk_filter
        return len_filter

    dataset = dataset.filter(
        filter_fn,
        input_columns=["smiles"],
        batched=True,
        batch_size=10000,
        num_proc=num_proc,
    )
    dataset.set_format("np")
    print("after filter:", dataset)
    dataset.save_to_disk(save_path, num_shards=num_shards, num_proc=num_proc)


def normalize_smiles(smiles: str, canonical: bool = True, isomeric: bool = False):
    try:
        res = Chem.MolToSmiles(
            Chem.MolFromSmiles(smiles),
            canonical=canonical,
            isomericSmiles=isomeric,
        )
    except:
        return ""
    return res


@app.command()
def canonicalize(
    dataset_path: Path,
    save_path: Path,
    canonical: bool = True,
    isomeric: bool = False,
    batch_size: int = 10000,
    num_proc: int = 4,
    num_shards: int = 10,
):

    dataset = datasets.load_from_disk(dataset_path)
    dataset.set_format("np")

    def canonicalize_fn(smiles_list):
        return {
            "smiles": np.array(
                [
                    normalize_smiles(x, canonical=canonical, isomeric=isomeric)
                    for x in smiles_list
                ]
            )
        }

    dataset = dataset.map(
        canonicalize_fn,
        input_columns=["smiles"],
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
    ).filter(
        lambda x: x != "",
        input_columns=["smiles"],
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
    )

    dataset.set_format("np")
    dataset.save_to_disk(save_path, num_shards=num_shards, num_proc=num_proc)


if __name__ == "__main__":
    app()
