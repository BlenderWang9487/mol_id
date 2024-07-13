# Mol ID

try to train a molecule SMILE transformer encoder with flash attention 2

## Environment

```bash
## Install torch via official command
# https://pytorch.org/get-started/locally/

## install packages
pip install lightning transformers datasets typer wandb rdkit

## Flash attention 2 is installed separately
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install flash-attn --no-build-isolation
```

## Model weight

Pretrain weight (trained on 50M zinc SMILES for 5 epoch) is freely available on huggingface

- [huggingface model page](https://huggingface.co/blenderwang/mol_id)

## Pretrain data

Pretrain dataset is also available on huggingface

- [huggingface dataset page](https://huggingface.co/datasets/blenderwang/zinc-50M)
    - [source data](https://files.docking.org/zinc20-ML/smiles/ZINC20_smiles_chunk_1.tar.gz)
    - Filtering step:
        - canonicalize (canonical: bool = True, isomeric: bool = False)
        - filter out those can't be canonicalized
        - filter out those has len > 512 or len < 8 or contains `<unk>` tokens
            - the properties mentioned here are all after tokenization

## Usage

See [example.py](example.py)
