
from string import Template
from gridsearcher import GridSearcher, SchedulingConfig, TorchRunConfig

def main(dist, gpus, params):
    gs = GridSearcher(script="/nfs/scistore19/alistgrp/imodoran/workplace/DAS-Standard/pretrain.py")
    gs.run(
        param_name_for_exp_root_folder="output_folder",
        exp_folder=Template("${run_name}"),
        cfg_torchrun=TorchRunConfig(
            launch_blocking=0,
            torchrun=dist,
            master_addr='127.0.0.1',
            master_port=29500 + gpus[0],  # heuristic
            rdzv_backend='static', # static, c10d
        ),
        cfg_sched=SchedulingConfig(
            distributed_training=True,
            max_jobs_per_gpu=1,
            gpus=gpus,
            params_values=params
        )
    )


if __name__ == "__main__":
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
            ### model section
            "model_name": ["llama"],
            # "vocab_size": [-1], # not needed, automatically determined
            "num_layers": [6],
            "num_heads": [8],
            "embed_size": [1024],
            "seq_len": [1024],
            "dropout": ["0.0"],
            "bias": [False],
            "tie_embeds": [False],
            "mlp_expansion_factor": [4],

            ### dataset section
            "dataset_path": ["./data"],
            "dataset_name": ["c4"],

            ### training section
            "seed": [42],
            "tokens_count": [0],
            "chinchilla_token_ratio": [1],
            "local_batch_size": [64],
            "grad_acc_steps": [8],
            # "max_iters": [-1], # not needed, automatically determined
            "backend": ["ddp"],
            "dtype": ["bf16"],
            "compile_model": [False],
            "eval_interval": [100],
            "eval_batches": [32],

            ### optimizer section
            "optimizer_name": ["adamw"],
            "weight_decay": ["0.0"],
            "grad_clipping": ["1.0"],

            # Learning Rate scheduling
            "lr_scheduler": ["cosine"],
            "lr_warmup": ["0.1"],
            "cos_start_lr_ratio": ["0.01"],
            "cos_end_lr_ratio": ["0.1"],

            # AdamW parameters
            "adamw_lr": ["1e-3"],
            "adamw_beta1": ["0.9"],
            "adamw_beta2": ["0.95"],
            "adamw_eps": ["1e-8"],

            # Muon parameters
            "muon_lr": ["1e-3"],
            "muon_momentum": ["0.95"],

            ### wandb section
            "entity": ["ist"],
            "project": ["das-standard"],
            "group": [Template("${model_name}_${optimizer_name}_wd=${weight_decay}_b=${adamw_beta1}-${adamw_beta2}_e=${adamw_eps}")],
            "job_type": [Template("lr=${adamw_lr}")],
            "run_name": [Template("${group}_${job_type}_seed=${seed}")],
        }
    )
