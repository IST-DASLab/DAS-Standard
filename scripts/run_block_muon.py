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

    gpus = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ]

    # dist = False
    dist = True

    optimizer = "adamw"
    # optimizer = "muon"
    # optimizer = "block_muon"

    match optimizer:
        case "adamw":
            wandb_group = Template("${model_name_size}_${optimizer_name}_wd=${weight_decay}_b=${adamw_beta1}-${adamw_beta2}_e=${adamw_eps}_sched=${"
                                   "lr_sched_name}")
        case "muon" | "block_muon":
            wandb_group = Template("${model_name_size}_${optimizer_name}_s=${muon_scaling_type}_wd=${weight_decay}_mu=${muon_momentum}_sched=${lr_sched_name}")
        case _:
            print(f"Unknown optimizer {optimizer}")
            exit(666)

    params = {
        ##########################
        ##### TrainingConfig #####
        ##########################
        "model_name_size": [
            # "gpt-42m",
            # "llama2-44m",
            # "llama-1b-custom",
            "llama-4b-custom",
            # "llama-8b-custom",
            # "qwen3-42m",
        ],
        "seed": [
            42,
            # 666,
            # 2408,
        ],
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
        "eval_interval": [
            # 100,
            # 500,
            1000,
        ],
        "eval_batches": [128],

        ################################################
        ##### Overrides to the default ModelConfig #####
        ################################################
        "seq_len": [1024],
        "vocab_size": [32_000],

        ###########################
        ##### OptimizerConfig #####
        ###########################
        "optimizer_name": [optimizer],
        "weight_decay": [
            "1e-1",
            # "1e-2",
        ],
        "grad_clipping": ["1.0"],

        # Learning Rate scheduling
        "lr_sched_name": [
            "cosine",
            # "const",
        ],
        "lr_sched_warmup": [
            # "1000",
            "2000",
        ], # actual number of steps instead of percentage
        # "lr_sched_wsd_decay": ["0.9"], # not used for const
        "lr_sched_start_ratio": ["0.01"],
        "lr_sched_end_ratio": ["0.1"], # not used for const

        # AdamW parameters
        "lr": [
            "1e-3",
            # "2e-3",
            # "3e-3",
            # "4e-3",

            # "5e-3",
            # "6e-3",
            # "7e-3",
            # "8e-3",
        ],
        "adamw_beta1": ["0.9"],
        "adamw_beta2": ["0.95"],
        "adamw_eps": ["1e-8"],

        # Muon parameters
        "muon_momentum": ["0.95"],
        "muon_scaling_type": [
            "rms",
            # "block",
        ],

        #########################
        ##### DatasetConfig #####
        #########################
        "dataset_path": [DATASETS_DIR],
        "dataset_name": ["c4"],

        #######################
        ##### WandbConfig #####
        #######################
        "entity": ["ionutmodo"],
        "project": ["ionut_block_muon"],
        "group": [wandb_group], # moved out
        "job_type": [
            Template("lr=${lr}"), # AdamW
        ],
        "run_name": [Template("${group}_${job_type}_seed=${seed}")],
    }

    main(
        dist=dist,
        gpus=gpus,
        params=params
    )
