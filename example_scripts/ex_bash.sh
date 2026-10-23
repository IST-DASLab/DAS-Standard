#!/bin/bash

MODEL=llama2-44m
OPTIMIZER=adamw
LR=1e-3
WD=1e-2
BETA1=0.9
BETA2=0.95
EPS=1e-8
SEED=42

WANDB_GROUP="${MODEL}_${OPTIMIZER}_wd=${WD}_b=${BETA1}-${BETA2}_e=${EPS}"
WANDB_JOB_TYPE="lr=${LR}"
WANDB_RUN_NAME="${WANDB_GROUP}_${WANDB_JOB_TYPE}_${SEED}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 pretrain.py \
    ### model section
    --seq_len 1024 \
    --weight_tying False \
    --gpt_mlp_exp_factor 4 \
    --attn_type mhsa-flash \

    ### dataset section
    --dataset_path ./data \
    --dataset_name c4 \

    ### training section
    # Checkpointing
    # --chkpt_resume_folder  \
    # --chkpt_resume_step 100 \
    --chkpt_max_limit 0 \
    --chkpt_freq 100 \

    --model_name_size $MODEL \
    --seed $SEED \
    --chinchilla_token_ratio 1 \
    --local_batch_size 64 \
    --grad_acc_steps 8 \
    --backend ddp \
    --amp_dtype bf16 \
    --compile_model \
    --eval_interval 100 \
    --eval_batches_per_worker 32 \

    ### optimizer section
    --optimizer_name adamw \
    --weight_decay $WD \
    --grad_clipping 1.0 \

    # Learning Rate scheduling
    --lr ${LR} \
    --lr_sched_name cosine \
    --lr_sched_warmup 0.1 \
    --lr_sched_wsd_decay 0.9 \
    --lr_sched_start_ratio 0.01 \
    --lr_sched_end_ratio 0.1 \

    # AdamW parameters
    --adamw_beta1 $BETA1 \
    --adamw_beta2 $BETA2 \
    --adamw_eps $EPS \

    # Muon parameters
    --muon_momentum 0.95 \
    --muon_scaling_type rms \

    ### QAT
    --qat_method quartet2 \
    --hadamard_size 128 \

    ### wandb section
    --entity your-wandb-entity \
    --project das-standard \
    --group ${WANDB_GROUP} \
    --job_type ${WANDB_JOB_TYPE} \
    --run_name ${WANDB_RUN_NAME}