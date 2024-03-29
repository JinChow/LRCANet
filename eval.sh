#!/usr/bin/env bash
set -x  # print the commands

export MASTER_PORT=${MASTER_PORT:-12321}  # You should set the same master_port in all the nodes


OUTPUT_DIR='/home/zj/Model_Code/new_FECN/results'
DATA_PATH='/home/zj/Dataset/Ekman6/list'
MODEL_PATH='/home/zj/Model_Code/new_FECN/train_result/Ekman6/checkpoint-best/mp_rank_00_model_states.pt'
N_NODES=1  # Number of nodes
GPUS_PER_NODE=1  # Number of GPUs in each node
SRUN_ARGS=${SRUN_ARGS:-""}  # Other slurm task args
PY_ARGS=${@:3}  # Other training args

# Please refer to `run_class_finetuning.py` for the meaning of the following hyperreferences
CUDA_VISIBLE_DEVICES="3" torchrun --nproc_per_node=1 --master_port=29514 \
       run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_set Ekman6 \
        --nb_classes 6 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 4 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 10 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 2 \
        --num_workers 14 \
        --opt adamw \
        --lr 4e-5 \
        --drop_path 0.3 \
        --clip_grad 5.0 \
        --layer_decay 0.9 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.1 \
        --warmup_epochs 0 \
        --epochs 100 \
        --test_num_segment 6 \
        --test_num_crop 3 \
        --dist_eval --enable_deepspeed \
        ${PY_ARGS}