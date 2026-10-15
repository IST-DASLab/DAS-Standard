import socket
import random
from datetime import datetime
from string import Template
from gridsearcher import GridSearcher, SchedulingConfig, TorchRunConfig, wait_for_resources

random.seed(datetime.now().timestamp())


def main(dist, gpus, params):
    PROJECT_FOLDER = "/localhome" if socket.gethostname() == "gpu413" else "/nfs/scistore19/alistgrp/"
    gs = GridSearcher(script=f"{PROJECT_FOLDER}/imodoran/workplace/DAS-Standard/pretrain.py")
    gs.run(
        param_name_for_exp_root_folder="output_folder",
        exp_folder=Template("results/${project}/${run_name}"),
        cfg_torchrun=TorchRunConfig(
            launch_blocking=0,
            torchrun=dist,
            master_addr='127.0.0.1',
            master_port=29000 + random.randint(0, 500),  # heuristic
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
    wait_for_resources()
    DATASETS_DIR = '/localhome/imodoran/datasets/' if socket.gethostname().startswith('gpu') else '/home/imodoran/datasets/'

    # OPTIMIZER = "adamw"
    OPTIMIZER = "trion"

    dist = False
    gpus = [
        0,
        1,
        3,
        6,

        # 0,
        # 1,
        # 2,
        # 3,

        # 4,
        # 5,
        # 6,
        # 7,
    ]

    params = {
        ##########################
        ##### TrainingConfig #####
        ##########################0
        "model_name_size": [
            # "gpt-42m",
            "llama2-44m",
            # "llama3-44m",
            # "qwen3-42m",
        ],
        "seed": [42],
        "chinchilla_token_ratio": [1],
        "local_batch_size": [16],
        "grad_acc_steps": [32],
        "backend": [
            "single",
            # "ddp",
            # "fsdp2",
        ],
        "amp_dtype": ["bf16"],
        "compile_model": [True],
        "eval_interval": [
            # 10,
            50,
            # 100,
        ],
        "eval_batches": [32],

        ################################################
        ##### Overrides to the default ModelConfig #####
        ################################################
        "seq_len": [1024],
        "vocab_size": [32_000],
        "dtype": [
            # "bf16",
            "fp32",
        ],

        #####################
        ##### QATConfig #####
        #####################
        "qat_method": [
            "disabled",
            # "quartet2",
        ],
        "hadamard_size": [128],

        ###########################
        ##### OptimizerConfig #####
        ###########################
        "optimizer_name": [OPTIMIZER],
        "weight_decay": ["0.0"],
        "grad_clipping": ["1.0"],

        # Learning Rate scheduling
        "lr_sched_name": ["cosine"],
        "lr_sched_warmup": ["0.1"],
        "lr_sched_start_ratio": ["0.01"],
        "lr_sched_end_ratio": ["0.1"],

        # AdamW parameters
        "adamw_lr": [
            "1e-3",
            "2e-3",
            "3e-3",
            "4e-3",
            "5e-3",
        ],
        "adamw_beta1": ["0.9"],
        "adamw_beta2": ["0.95"],
        "adamw_eps": ["1e-8"],

        # Trion parameters
        "trion_lr": [
            "1e-2",
            "2e-2",
            "3e-2",
            "4e-2",
            "5e-2",
        ],
        "trion_momentum": ["0.95"],
        "trion_rank": [
            # 1,
            # 2,
            # 4,
            8,

            # 64, # 12.5%
            # 128, # 25%
            # 256, # 50%
        ],
        "trion_scaling": [
            "kj",
            # "none",
            # "rms",
            # "dion",
        ],
        "trion_proj": [
            "dct",
            "eye",
        ],

        #########################
        ##### DatasetConfig #####
        #########################
        "dataset_path": [DATASETS_DIR],
        "dataset_name": ["c4"],

        #######################
        ##### WandbConfig #####
        #######################
        "entity": ["ist"],
        "project": [f"das-standard-trion-{socket.gethostname()}"],
        "group": ["placeholder"],
        "job_type": ["placeholder"],
        "run_name": [Template("${group}_${job_type}_seed=${seed}")],
    }

    match OPTIMIZER:
        case "adamw":
            params["group"] = [Template("${model_name_size}_${optimizer_name}")]
            params["job_type"] = [Template("lr=${adamw_lr}")]
        case "trion":
            params["group"] = [
                Template(
                    "${model_name_size}_"
                    "${optimizer_name}_"
                    "r=${trion_rank}_"
                    "s=${trion_scaling}_"
                    "p=${trion_proj}"),
            ]
            params["job_type"] = [Template("lr-adamw=${adamw_lr}_lr-trion=${trion_lr}")]
        case _:
            raise RuntimeError(f"Unknown OPTIMIZER: {OPTIMIZER}")

    main(dist=dist, gpus=gpus, params=params)
