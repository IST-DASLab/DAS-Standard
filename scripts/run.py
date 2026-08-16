import socket
from string import Template
from gridsearcher import GridSearcher, SchedulingConfig, TorchRunConfig

def main(dist, gpus, params):
    gs = GridSearcher(script="/nfs/scistore19/alistgrp/imodoran/workplace/DAS-Standard/pretrain.py")
    gs.run(
        param_name_for_exp_root_folder="output_folder",
        exp_folder=Template("results/${run_name}"),
        cfg_torchrun=TorchRunConfig(
            launch_blocking=0,
            torchrun=dist,
            master_addr='127.0.0.1',
            master_port=29500 + gpus[0],  # heuristic
            rdzv_backend='static' if 'gpu413' in socket.gethostname() else 'c10d',
        ),
        cfg_sched=SchedulingConfig(
            distributed_training=True,
            max_jobs_per_gpu=1,
            gpus=gpus,
            params_values=params
        )
    )

if __name__ == "__main__":
    DATASETS_DIR = '/localhome/imodoran/datasets/' if socket.gethostname().startswith('gpu') else '/home/imodoran/datasets/'

    main(
        dist=True,
        gpus=[
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
        ],
        params={
            ##########################
            ##### TrainingConfig #####
            ##########################
            "model_name_size": ["llama2"],
            "seed": [42],
            # "num_tokens": [0], # lower priority over chinchilla_token_ratio
            "chinchilla_token_ratio": [1],
            "local_batch_size": [16],
            "grad_acc_steps": [32],
            # "max_iters": [-1], # not needed, automatically determined
            "backend": ["ddp"],
            "amp_dtype": ["bf16"],
            "compile_model": [True],
            "eval_interval": [100],
            "eval_batches": [32],



            #####################
            ##### QATConfig #####
            #####################
            "qat_method": ["disabled"],
            "hadamard_size": [128],



            ###########################
            ##### OptimizerConfig #####
            ###########################
            "optimizer_name": [
                "adamw",
                # "muon",
            ],
            "weight_decay": ["0.0"],
            "grad_clipping": ["1.0"],

            # Learning Rate scheduling
            "lr_sched_name": ["cosine"],
            "lr_sched_warmup": ["0.1"],
            "lr_sched_start_ratio": ["0.01"],
            "lr_sched_end_ratio": ["0.1"],

            # AdamW parameters
            "adamw_lr": ["1e-3"],
            "adamw_beta1": ["0.9"],
            "adamw_beta2": ["0.95"],
            "adamw_eps": ["1e-8"],

            # Muon parameters
            "muon_lr": ["4e-2"],
            "muon_momentum": ["0.95"],



            #########################
            ##### DatasetConfig #####
            #########################
            "dataset_path": [DATASETS_DIR],
            "dataset_name": ["c4"],



            #######################
            ##### WandbConfig #####
            #######################
            "entity": ["ist"],
            "project": ["das-standard"],
            "group": [
                # Template("${model_name_size}_${optimizer_name}_wd=${weight_decay}_b=${adamw_beta1}-${adamw_beta2}_e=${adamw_eps}"), # AdamW
                Template("${model_name_size}_${optimizer_name}_wd=${weight_decay}_mu=${muon_momentum}"), # Muon
            ],
            "job_type": [
                # Template("lr=${adamw_lr}"), # AdamW
                Template("lr-muon=${muon_lr}_lr-adamw=${adamw_lr}"),  # Muon
            ],
            "run_name": [Template("${group}_${job_type}_seed=${seed}")],
        }
    )
