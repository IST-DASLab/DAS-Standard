import socket
import random
from datetime import datetime
from string import Template
from gridsearcher import GridSearcher, SchedulingConfig, TorchRunConfig

random.seed(datetime.now().timestamp())

def main(dist, gpus, params):
    gs = GridSearcher(script="/nfs/scistore19/alistgrp/imodoran/workplace/DAS-Standard/pretrain.py")
    gs.run(
        param_name_for_exp_root_folder="output_folder",
        exp_folder=Template("/mnt/beegfs/alistgrp/imodoran/results/${project}/${run_name}"),
        cfg_torchrun=TorchRunConfig(
            launch_blocking=0,
            torchrun=dist,
            master_addr='127.0.0.1',
            master_port=29000 + random.randint(0,500),  # heuristic
            rdzv_backend='static' if 'gpu413' in socket.gethostname() else 'c10d',
        ),
        cfg_sched=SchedulingConfig(
            distributed_training=dist,
            max_jobs_per_gpu=1,
            gpus=gpus,
            params_values=params
        )
    )

if __name__ == "__main__":
    DATASETS_DIR = '/localhome/imodoran/datasets/' if socket.gethostname().startswith('gpu') else '/home/imodoran/datasets/'

    OPTIMIZER = "adamw"

    main(
        # dist=False,
        dist=True,
        gpus=[
            # 0,
            # 1,
            # 2,
            # 3,
            # 4,
            # 5,
            6,
            # 7,
        ],
        params={
            ##########################
            ##### Checkpointing  #####
            ##########################
            "chkpt_resume_folder": [
                '/mnt/beegfs/alistgrp/imodoran/results/das-standard-test/single-chkpt_llama2-44m_adamw_lr=1e-3_seed=42',
                # '/mnt/beegfs/alistgrp/imodoran/results/das-standard-test/ddp-chkpt_llama2-44m_adamw_lr=1e-3_seed=42',
                # '/mnt/beegfs/alistgrp/imodoran/results/das-standard-test/fsdp2-chkpt_llama2-44m_adamw_lr=1e-3_seed=42',
            ],
            "chkpt_resume_step": [100],
            "chkpt_max_limit": [0],
            "chkpt_freq": [100],

            ##########################
            ##### ModelConfig    #####
            ##########################
            "attn_type": [
                # "mhsa",
                "mhsa-flash",
                # "gqa",
                # "gqa-flash",
            ],

            ##########################
            ##### TrainingConfig #####
            ##########################
            "model_name_size": [
                # "gpt-42m",
                "llama2-44m",
                # "llama-1b-custom",
                # "llama3-44m",
                # "qwen3-42m",
            ],
            "seed": [42],
            # "num_tokens": [0], # lower priority over chinchilla_token_ratio
            "chinchilla_token_ratio": [1],
            "local_batch_size": [16],
            "grad_acc_steps": [32],
            # "max_iters": [-1], # not needed, automatically determined
            "backend": [
                # "single",
                # "ddp",
                "fsdp2",
            ],
            "amp_dtype": ["bf16"],
            "compile_model": [True],
            "eval_interval": [50],
            "eval_batches_per_worker": [32],

            ################################################
            ##### Overrides to the default ModelConfig #####
            ################################################
            "seq_len": [1024],
            "vocab_size": [32_000],

            ###########################
            ##### OptimizerConfig #####
            ###########################
            "optimizer_name": ["adamw"],
            "weight_decay": ["0.0"],
            "grad_clipping": ["1.0"],

            # Learning Rate scheduling
            "lr_sched_name": ["cosine"],
            "lr_sched_warmup": ["0.1"],
            "lr_sched_start_ratio": ["0.01"],
            "lr_sched_end_ratio": ["0.1"],

            # AdamW parameters
            "lr": [
                "1e-3",
            ],
            "adamw_beta1": ["0.9"],
            "adamw_beta2": ["0.95"],
            "adamw_eps": ["1e-8"],

            #########################
            ##### DatasetConfig #####
            #########################
            "dataset_path": [DATASETS_DIR],
            "dataset_name": ["c4"],

            #######################
            ##### WandbConfig #####
            #######################
            "entity": ["ist"],
            "project": [f"das-standard-test"],
            "group": [
                Template(
                    # "single-chkpt_${model_name_size}_${optimizer_name}"
                    "single-chkpt-load-${resume_step}_${model_name_size}_${optimizer_name}"

                    # "ddp-chkpt_${model_name_size}_${optimizer_name}"
                    # "ddp-chkpt-load-${resume_step}_${model_name_size}_${optimizer_name}"

                    # "fsdp2-chkpt_${model_name_size}_${optimizer_name}"
                    # "fsdp2-chkpt-load-${resume_step}_${model_name_size}_${optimizer_name}"
                )
            ],
            "job_type": [
                Template("lr=${lr}"),
            ],
            "run_name": [Template("${group}_${job_type}_seed=${seed}")],
        }
    )
