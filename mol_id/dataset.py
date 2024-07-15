from functools import partial
from pathlib import Path

import datasets
import numpy as np
import typer
from rdkit import Chem

from .molformer_tokenizer import MolTranBertTokenizer


def normalize_smiles(
    smiles: str, canonical: bool = True, isomeric: bool = False
) -> str | None:
    try:
        res = Chem.MolToSmiles(
            Chem.MolFromSmiles(smiles),
            canonical=canonical,
            isomericSmiles=isomeric,
        )
    except:
        return None
    return res


def filter_func(
    smiles: str,
    tokenizer: MolTranBertTokenizer,
    unk_id: int,
    max_len: int = 512,  # after tokenization
    min_len: int = 8,  # after tokenization
    canonical: bool = True,
    isomeric: bool = False,
    filter_unk: bool = True,
):
    smiles = normalize_smiles(smiles, canonical, isomeric)
    if smiles is None:
        return None
    tok_output = tokenizer.encode_plus(
        smiles,
        return_token_type_ids=False,
        return_attention_mask=False,
        return_length=True,
        return_tensors="np",
    )
    l = tok_output["length"][0]
    if l > max_len or l < min_len:
        return None
    if filter_unk and (unk_id in tok_output["input_ids"]):
        return None
    return smiles


def file_iterator(files: list[Path], smiles_filter):
    for file in files:
        with open(file, "r") as f:
            for line in f:
                smiles: str | None = smiles_filter(line.split(maxsplit=1)[0])
                if smiles is None:
                    continue
                yield {"smiles": smiles}


app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def build(
    data_path: Path,
    save_path: Path,
    max_len: int = 512,  # after tokenization
    min_len: int = 8,  # after tokenization
    canonical: bool = True,
    isomeric: bool = False,
    filter_unk: bool = True,
    num_proc: int = 4,
    num_shards: int = 10,
    cache_dir: Path = None,
):
    files = list(data_path.glob("*.txt")) if data_path.is_dir() else [data_path]
    tok = MolTranBertTokenizer()
    unk_id = tok.unk_token_id
    print("### unk_id:", unk_id)
    composed_filter = partial(
        filter_func,
        tokenizer=tok,
        unk_id=unk_id,
        max_len=max_len,
        min_len=min_len,
        canonical=canonical,
        isomeric=isomeric,
        filter_unk=filter_unk,
    )
    composed_iterator = partial(file_iterator, smiles_filter=composed_filter)

    dataset = datasets.Dataset.from_generator(
        composed_iterator,
        gen_kwargs={
            "files": files,
        },
        num_proc=num_proc,
        cache_dir=cache_dir,
    )

    dataset.set_format("np")
    dataset.save_to_disk(save_path, num_shards=num_shards, num_proc=num_proc)


@app.command()
def placeholder():
    pass


if __name__ == "__main__":
    app()
