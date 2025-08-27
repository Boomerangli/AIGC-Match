#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['acdc']
# method: ['unimatch', 'supervised']
# exp: just for specifying the 'save_path'
# split: ['1', '3', '7']
dataset='acdc'
method='test2'
exp='unet'
split='7'

config=configs/$dataset.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt


python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path 
