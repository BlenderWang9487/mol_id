set -ex

WKEY=$1
HKEY=$2

pip install lightning transformers datasets typer wandb

## only for dataset building or SMILES augmentation
# pip install rdkit

## Flash attention 2 is installed separately
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install flash-attn --no-build-isolation

wandb login $WKEY
huggingface-cli login --token $HKEY