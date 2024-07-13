# THIS SCRIPT IS JUST FOR DEVELOPMENT CONVINIENCE
# YOU CAN IGNORE THIS
#
visible=${1:-0,1}
conda activate molclip
set -ex
export CUDA_VISIBLE_DEVICES=$visible
set +ex