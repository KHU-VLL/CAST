#!/bin/bash

DATA_PATH=YOUR_PATH
VMAE_MODEL_PATH=YOUR_PATH
CLIP_MODEL_PATH=YOUR_PATH

OUTPUT_DIR=YOUR_PATH
MASTER_NODE=$1
OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=$6 \
    --master_port $3 --nnodes=$5 \
    --node_rank=$2 --master_addr=${MASTER_NODE} \
    YOUR_PATH/run_bidirection.py \
    --data_set Kinetics-400 \
    --nb_classes 400 \
    --vmae_model bidir_vit_base_patch16_224 \
    --anno_path ${ANNOTATION_PATH} \
    --data_path ${DATA_PATH} \
    --clip_finetune ${CLIP_MODEL_PATH} \
    --vmae_finetune ${VMAE_MODEL_PATH} \
    --log_dir ${YOUR_PATH} \
    --output_dir ${YOUR_PATH} \
    --batch_size 6 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 25 \
    --num_sample 1 \
    --num_frames 16 \
    --opt adamw \
    --lr 1e-3 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 70 \
    --dist_eval \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --num_workers 8 \
    --drop_path 0.2 \
    --layer_decay 0.75 \
    --mixup_switch_prob 0 \
    --mixup_prob 0.5 \
    --reprob 0. \
    --init_scale 1. \
    --update_freq 6 \
    --seed 0 \
    --enable_deepspeed \
    --warmup_epochs 5 \

echo "Job finish"