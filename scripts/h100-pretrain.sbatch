#!/bin/bash
#ASC --vanilla
#SBATCH --account=p201397
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive # uses all available CPUs (instead of specifying --cpus-per-task=128)
#SBATCH --threads-per-core=1
#SBATCH --time=3-00:00:00
#SBATCH --nodes=2
#SBATCH --job-name=muon
#SBATCH --output=/data/fs201397/im31873/logs/%x-%j-DATETIME.out
#SBATCH --error=/data/fs201397/im31873/logs/%x-%j-DATETIME.err

set -e
source ~/.bashrc
conda activate "$DATA/envs/das-standard"

rdzv_host=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

srun --ntasks="$SLURM_JOB_NUM_NODES" --cpus-per-task="$SLURM_CPUS_PER_TASK" --gpu-bind=none \
    torchrun \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc-per-node=gpu \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${rdzv_host}:29062" \
    --rdzv-id="$SLURM_JOB_ID" \
    /data/fs201397/im31873/projects/DAS-Standard/pretrain.py --model_name_size llama-1b-custom --seed 42 --chinchilla_token_ratio 1 --local_batch_size 32 --grad_acc_steps 16 --backend fsdp2 --compile_model --eval_interval 1000 --eval_batches_per_worker 512 --chkpt_max_limit 0 --chkpt_freq 5000 --seq_len 1024 --rope_theta 10000 --vocab_size 32000 --storage_dtype fp32 --compute_dtype bf16 --fsdp_reduce_dtype fp32 --qat_method disabled --hadamard_size 128 --optimizer_name block_muon_head --weight_decay 1e-1 --weight_decay_non_linear 1e-1 --grad_clipping 1.0 --lr_sched_name cosine --lr_sched_warmup 0.1 --lr_sched_start_ratio 0.01 --lr_sched_end_ratio 0.1 --lr 1e-3 --adamw_beta1 0.9 --adamw_beta2 0.95 --adamw_eps 1e-8 --muon_momentum 0.95 --muon_scaling_type block --muon_num_blocks 0 --muon_coeffs_type p5-polar --dataset_path /scratch/fs201397/im31873/datasets --dataset_name c4 --entity ionutmodo --project ionut_block_muon_reviewed --group pc=fp32-bf16-fp32_llama-1b-custom_block_muon_head_wd=1e-1-1e-1_s=block_nB=0_c=p5-polar_sched=cosine --job_type lr=1e-3 --run_name pc=fp32-bf16-fp32_llama-1b-custom_block_muon_head_wd=1e-1-1e-1_s=block_nB=0_c=p5-polar_sched=cosine_lr=1e-3_seed=42 --output_folder /scratch/fs201397/im31873/results/ionut_block_muon_reviewed/pc=fp32-bf16-fp32_llama-1b-custom_block_muon_head_wd=1e-1-1e-1_s=block_nB=0_c=p5-polar_sched=cosine_lr=1e-3_seed=42

# sed "s/DATETIME/$(date +%Y-%m-%d_%H-%M-%S)/g" pretrain.sbatch | sbatch