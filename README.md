# DAS-Standard
This is a self-sufficient repository for pretraining experiments.

## Usage

### Create environment
```bash
conda create --name das_standard python=3.12 -c conda-forge --override-channels -y
conda activate das_standard
```

### Clone repository & install requirements
```bash
cd /tmp
git clone git@github.com:IST-DASLab/DAS-Standard.git
cd /tmp/DAS-Standard
pip install -r requirements.txt
```

### Download dataset
```bash
cd /tmp/DAS-Standard
python download_hf_data.py # modify `local_dir` in this file if needed
cd /tmp/DAS-Standard/datasets
cat chunk_* > train.bin # merge training chunks into a single training file
```

### Run example script using GridSearcher
```bash
cd /tmp/DAS-Standard
python example_scripts/ex_gridsearcher.py # edit `dataset_path` argument or `script_path` in GridSearcher constructor
```

### Example output:
```text
GridSearcher PID: 1736237
Commands:	Runnable: 1	Finished: 0	Total: 1
Waiting 5 seconds before running GridSearcher...

=== Using FSDP2 backend ===

==================================================
                  EXPERIMENT CONFIGURATION                  
==================================================
[DATASET]
  dataset_path              : /tmp/DAS-Standard/datasets/
  dataset_name              : c4

[MODEL]
  dtype                     : torch.bfloat16
  attn_type                 : mhsa
  attn_gqa_kvg              : 8
  qk_norm                   : 0
  rope_theta                : 10_000.0
  vocab_size                : 32_000
  num_layers                : 8
  num_heads                 : 8
  embed_size                : 512
  seq_len                   : 1_024
  dropout                   : 0.0
  bias                      : 0
  tie_embeds                : 1
  gpt_mlp_exp_factor        : 4
  llama_mlp_size            : 1_536

[OPTIMIZER]
  optimizer_name            : adamw
  weight_decay              : 0.0
  grad_clipping             : 1.0
  lr_sched_name             : cosine
  lr_sched_warmup           : 0.1
  lr_sched_start_ratio      : 0.01
  lr_sched_end_ratio        : 0.1
  adamw_lr                  : 0.001
  adamw_beta1               : 0.9
  adamw_beta2               : 0.95
  adamw_eps                 : 1e-08
  muon_lr                   : 0.04
  muon_momentum             : 0.95
  microadam_m               : 10
  microadam_q_block         : 100_000
  microadam_k               : 0.01

[QAT]
  qat_method                : quartet2
  hadamard_size             : 128

[TRAINING]
  model_name_size           : llama2-44m
  seed                      : 42
  num_tokens                : 545_423_380
  chinchilla_token_ratio    : 1
  local_batch_size          : 16
  grad_acc_steps            : 4
  max_iters                 : 1_041
  backend                   : fsdp2
  amp_dtype                 : bf16
  compile_model             : 1
  eval_interval             : 10
  eval_batches              : 32
  output_folder             : /tmp/DAS-Standard/results/qat=quartet2_H=128_llama2-44m_adamw_wd=0.0_b=0.9-0.95_e=1e-8_lr=1e-3_seed=42

[WANDB]
  entity                    : ist
  project                   : das-standard
  group                     : qat=quartet2_H=128_llama2-44m_adamw_wd=0.0_b=0.9-0.95_e=1e-8
  job_type                  : lr=1e-3
  run_name                  : qat=quartet2_H=128_llama2-44m_adamw_wd=0.0_b=0.9-0.95_e=1e-8_lr=1e-3_seed=42

==================================================
Model:
Llama(
  (token_embed): Embedding(32000, 512)
  (blocks): ModuleList(
    (0-7): 8 x LlamaBlock(
      (attn): MultiHeadSelfAttention(
        (Wq): Quartet_II_Linear(in_features=512, out_features=512, bias=False)
        (Wk): Quartet_II_Linear(in_features=512, out_features=512, bias=False)
        (Wv): Quartet_II_Linear(in_features=512, out_features=512, bias=False)
        (Wo): Quartet_II_Linear(in_features=512, out_features=512, bias=False)
        (dropout_attn): Dropout(p=0.0, inplace=False)
        (dropout_out): Dropout(p=0.0, inplace=False)
      )
      (mlp): LlamaFeedForward(
        (fc1): Quartet_II_Linear(in_features=512, out_features=1536, bias=False)
        (fc2): Quartet_II_Linear(in_features=512, out_features=1536, bias=False)
        (fc3): Quartet_II_Linear(in_features=1536, out_features=512, bias=False)
        (silu): SiLU()
      )
      (norm1): RMSNorm()
      (norm2): RMSNorm()
    )
  )
  (norm): RMSNorm()
  (lm_head): Linear(in_features=512, out_features=32000, bias=False)
)
==================================================
Model summary:
	 Input Embedding params: 16_384_000 (tied: True)
	Output Embedding params: 16_384_000 (tied: True)
	          Linear params: 27_262_976
	     Layer norms params: 8_704
	             All params: 43_655_680
==================================================
Total tokens & training steps:
	               Linear params (P): 27_262_976
	                 Norm params (N): 8_704
	 Total params for tokens (P + N): 27_271_680
	    Chinchilla Token Ratio (CTR): 20x
	      Tokens (T = CTR * (P + N)): 545_433_600
	             Sequence Length (S): 1024
	           Device Batch Size (B): 16
	              Grad Acc Steps (A): 4
	                  World Size (W): 8
	     Steps (T / (B * S * A * W)): 1_041
==================================================
Dataset tokens: train=197_980_429_805 | val=99_126_266
Training:   0%|                                                   | 0/1041 [00:00<?, ?it/s]
Evaluation: step=0 | num_batches=32 | loss=10.5625 | ppl=38657.6514
Training:   1%|█▎                                                  | 9/1041 [00:46<1:26:21,  5.02s/it, loss=9.8594]
Evaluation: step=10 | num_batches=32 | loss=9.6348 | ppl=15287.1154
Training:   2%|██▋                                                | 19/1041 [01:05<56:13,  3.30s/it, loss=8.9688]
Evaluation: step=20 | num_batches=32 | loss=8.7344 | ppl=6212.8500
Training:   3%|████                                               | 29/1041 [01:24<46:42,  2.77s/it, loss=7.9688]
Evaluation: step=30 | num_batches=32 | loss=7.8027 | ppl=2447.2846
Training:   4%|█████▍                                             | 39/1041 [01:43<41:52,  2.51s/it, loss=7.4531]
Evaluation: step=40 | num_batches=32 | loss=7.4219 | ppl=1672.1659
Training:   5%|██████▊                                           | 49/1041 [02:02<38:59,  2.36s/it, loss=7.2891]
Evaluation: step=50 | num_batches=32 | loss=7.2588 | ppl=1420.5353
Training:   6%|████████▎                                          | 59/1041 [02:21<36:58,  2.26s/it, loss=7.1406]
Evaluation: step=60 | num_batches=32 | loss=7.0498 | ppl=1152.6336
Training:   7%|█████████▋                                        | 69/1041 [02:40<35:30,  2.19s/it, loss=6.9297]
Evaluation: step=70 | num_batches=32 | loss=6.8682 | ppl=961.1823
Training:   8%|███████████                                       | 79/1041 [02:59<34:17,  2.14s/it, loss=6.6328]
Evaluation: step=80 | num_batches=32 | loss=6.5508 | ppl=699.7907
Training:   9%|████████████▍                                      | 89/1041 [03:18<33:21,  2.10s/it, loss=6.3906]
Evaluation: step=90 | num_batches=32 | loss=6.3525 | ppl=573.9482
Training:  10%|█████████████▉                                    | 99/1041 [03:37<32:30,  2.07s/it, loss=6.2656]
Evaluation: step=100 | num_batches=32 | loss=6.1982 | ppl=491.8836
Training:  10%|███████████████▏                                   | 109/1041 [03:56<31:47,  2.05s/it, loss=6.1250]
Evaluation: step=110 | num_batches=32 | loss=6.1074 | ppl=449.1792
Training:  11%|████████████████▌                                  | 119/1041 [04:15<31:07,  2.03s/it, loss=6.0000]
Evaluation: step=120 | num_batches=32 | loss=5.9727 | ppl=392.5470
Training:  12%|█████████████████▉                                 | 129/1041 [04:34<30:31,  2.01s/it, loss=5.8516]
Evaluation: step=130 | num_batches=32 | loss=5.8584 | ppl=350.1629
Training:  13%|███████████████████▎                                | 139/1041 [04:54<30:00,  2.00s/it, loss=5.8047]
Evaluation: step=140 | num_batches=32 | loss=5.7832 | ppl=324.7979
Training:  14%|████████████████████▊                              | 149/1041 [05:13<29:31,  1.99s/it, loss=5.7344]
Evaluation: step=150 | num_batches=32 | loss=5.7002 | ppl=298.9258
Training:  15%|██████████████████████▏                             | 159/1041 [05:32<29:02,  1.98s/it, loss=5.6172]
Evaluation: step=160 | num_batches=32 | loss=5.6104 | ppl=273.2403
Training:  16%|██████████████████████▌                            | 162/1041 [05:39<29:06,  1.99s/it, loss=5.6172]
```