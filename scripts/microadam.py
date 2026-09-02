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

    OPTIMIZER = "adamw"
    # OPTIMIZER = "microadam-torch"

    dist = False
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
        # "num_tokens": [0], # lower priority over chinchilla_token_ratio
        "chinchilla_token_ratio": [1],
        "local_batch_size": [16],
        "grad_acc_steps": [32],
        # "max_iters": [-1], # not needed, automatically determined
        "backend": [
            "single",
            # "ddp",
            # "fsdp2",
        ],
        "amp_dtype": ["bf16"],
        "compile_model": [
            0,
            1,
        ],
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
            # "1e-4",
            # "2e-4",
            # "4e-4",
            # "6e-4",
            # "8e-4",

            "1e-3",
            "2e-3",
            "4e-3",
            "6e-3",
            "8e-3",
        ],
        "adamw_beta1": ["0.9"],
        "adamw_beta2": ["0.95"],
        "adamw_eps": ["1e-8"],

        # Muon parameters
        "muon_lr": ["4e-2"],
        "muon_momentum": ["0.95"],

        # MicroAdam parameters
        "microadam_m": [
            # 1,
            # 5,
            10,
        ],
        "microadam_k": [
            "0.01",
            # "0.02",
            # "0.10",
        ],
        "microadam_ef_num_bits": [4],
        "microadam_q_block": [
            32,
            # 100_000,
        ],
        "microadam_dtype_scales": [
            "bf16",
            # "fp16",
        ],
        "microadam_dtype_values": [
            "bf16",
            # "fp16",
        ],
        "microadam_structured": [
            # 0,
            1,
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
        "project": [f"das-standard-microadam-{socket.gethostname()}"],
        "group": ["placeholder"],
        "job_type": [Template("lr=${adamw_lr}")],
        "run_name": [Template("${group}_${job_type}_seed=${seed}")],
    }

    match OPTIMIZER:
        case "adamw":
            params["group"] = [
                Template("compile=${compile_model}_${model_name_size}_${optimizer_name}")
            ]
        case "microadam-torch":
            params["group"] = [
                Template(
                    "compile=${compile_model}_"
                    "${model_name_size}_"
                    "${optimizer_name}_"
                    "m=${microadam_m}_"
                    "k=${microadam_k}_"
                    "ef-v=${microadam_dtype_scales}-${microadam_dtype_values}_"
                    "info=${microadam_quant_info}_"
                    "struc=${microadam_structured}"),
            ]
        case _:
            raise RuntimeError(f"Unknown OPTIMIZER: {OPTIMIZER}")

    main(dist=dist, gpus=gpus, params=params)
