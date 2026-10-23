import socket
import random
from datetime import datetime
from string import Template
from gridsearcher import GridSearcher, SchedulingConfig, TorchRunConfig

random.seed(datetime.now().timestamp())

def main(dist, gpus, params):
    gs = GridSearcher(script="/tmp/DAS-Standard/pretrain.py")
    gs.run(
        param_name_for_exp_root_folder="output_folder",
        exp_folder=Template("/tmp/DAS-Standard/results/${project}/${run_name}"),
        cfg_torchrun=TorchRunConfig(
            launch_blocking=0,
            torchrun=dist,
            master_addr='127.0.0.1',
            master_port=29000 + random.randint(0,500),  # heuristic
            rdzv_backend='c10d',
            # rdzv_backend='static',
        ),
        cfg_sched=SchedulingConfig(
            distributed_training=dist,
            max_jobs_per_gpu=1,
            gpus=gpus,
            params_values=params
        )
    )

if __name__ == "__main__":
    main(
        # dist=False,
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
            "local_batch_size": [16], # batch size per device
            "grad_acc_steps": [32],
            # "max_iters": [-1], # not needed, automatically determined
            "backend": [
                # "single",
                "ddp",
                # "fsdp2",
            ],
            "amp_dtype": ["bf16"],
            "compile_model": [True],
            "eval_interval": [100],
            "eval_batches_per_worker": [32], # all workers perform eval using this many batches, with "local_batch_size" in each batch

            ### Checkpointing:
            ### initial run: comment out chkpt_resume_folder and chkpt_resume_step such that they are None in the config
            ### resuming: provide the root path and specify chkpt_resume_step to load from

            # "chkpt_resume_folder": ["/tmp/DAS-Standard/results/your_project/your_run_name"],
            # "chkpt_resume_step": [100],
            "chkpt_max_limit": [0], # never delete old chkpts
            "chkpt_freq": [100], # log once at 100 steps


            ################################################
            ##### Overrides to the default ModelConfig #####
            ################################################
            "seq_len": [1024],
            "vocab_size": [32_000],



            #####################
            ##### QATConfig #####
            #####################
            "qat_method": [
                # "disabled",
                "quartet2",
            ],
            "hadamard_size": [
                # 32,
                # 64,
                128,
            ],



            ###########################
            ##### OptimizerConfig #####
            ###########################
            "optimizer_name": [
                "adamw",
                # "muon",
            ],
            "weight_decay": ["1e-2"],
            "grad_clipping": ["1.0"],

            # Learning Rate scheduling
            "lr": ["1e-3"],

            "lr_sched_name": [
                "cosine",
                # "linear",
                # "wsd",
                # "const",
            ],
            "lr_sched_warmup": ["0.1"], # float in (0,1) for percentage and int in [1, +oo) for actual steps
            "lr_sched_wsd_decay": ["0.9"], # decay after 90%
            "lr_sched_start_ratio": ["0.01"],
            "lr_sched_end_ratio": ["0.1"],

            # AdamW parameters
            "adamw_beta1": ["0.9"],
            "adamw_beta2": ["0.95"],
            "adamw_eps": ["1e-8"],

            # Muon parameters
            "muon_momentum": ["0.95"],
            "muon_scaling_type": [ # see function "muon_scaling" from muon_utils.py
                # "none",
                # "kj",
                "rms", # allows using the same lr from AdamW
                # "block",
                # "dion",
            ],



            #########################
            ##### DatasetConfig #####
            #########################
            "dataset_path": ["/tmp/DAS-Standard/datasets"],
            "dataset_name": ["c4"],



            #######################
            ##### WandbConfig #####
            #######################
            "entity": ["your_wandb_entity"],
            "project": ["das-standard"],
            "group": [ # here you can define a custom template for the wandb group
                ### AdamW
                Template("qat=${qat_method}_H=${hadamard_size}_${model_name_size}_${optimizer_name}_wd=${weight_decay}_b=${adamw_beta1}-${adamw_beta2}_e=${adamw_eps}"),

                ### Muon
                # Template("qat=${qat_method}_${model_name_size}_${optimizer_name}_wd=${weight_decay}_mu=${muon_momentum}"),
            ],
            "job_type": [
                Template("lr=${lr}"),
            ],
            "run_name": [Template("${group}_${job_type}_seed=${seed}")],
        }
    )
