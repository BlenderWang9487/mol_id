visible=${1:-0,1}
conda activate molclip
set -ex
export CUDA_VISIBLE_DEVICES=$visible
set +ex