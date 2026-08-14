#!/bin/bash

MODEL=llama
OPTIMIZER=adamw
ADAMW_LR=1e-3
WD=0
BETA1=0.9
BETA2=0.95
EPS=1e-8
SEED=42
WANDB_GROUP="${MODEL}_${OPTIMIZER}_wd=${WD}_b=${BETA1}-${BETA2}_e=${EPS}"
WANDB_JOB_TYPE="lr=${ADAMW_LR}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 pretrain.py \
    ### model section
    --model_name $MODEL \
    --num_layers 6 \
    --num_heads 8 \
    --embed_size 1024 \
    --seq_len 1024 \
    --dropout 0.0 \
    --bias False \
    --weight_tying False \
    --gpt_mlp_exp_factor 4 \

    ### dataset section
    --dataset_path ./data \
    --dataset_name c4 \

    ### training section
    --seed $SEED \
    --tokens_count 0 \
    --chinchilla_token_ratio 1 \
    --local_batch_size 64 \
    --grad_acc_steps 8 \
    --backend ddp \
    --dtype bf16 \
    --compile_model False \
    --eval_interval 100 \
    --eval_batches 32 \
    --log_interval 1 \

    ### optimizer section
    --optimizer_name adamw \
    --weight_decay $WD \
    --grad_clipping 1.0 \

    # Learning Rate scheduling
    --lr_scheduler cosine \
    --lr_warmup 0.1 \
    --cos_start_lr_ratio 0.01 \
    --cos_end_lr_ratio 0.1 \

    # AdamW parameters
    --adamw_lr ${ADAMW_LR} \
    --adamw_beta1 $BETA1 \
    --adamw_beta2 $BETA2 \
    --adamw_eps $EPS \

    # Muon parameters
    --muon_lr 1e-3 \
    --muon_momentum 0.95 \

    ### wandb section
    --entity ist \
    --project das-standard \
    --group ${WANDB_GROUP} \
    --job_type ${WANDB_JOB_TYPE} \
    --run_name ${WANDB_RUN_NAME}