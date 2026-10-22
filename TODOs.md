### TODO List for DAS-Standard

---

### Python Environment
- **[done]** requirements file

### Command Line Arguments
- **[done]** configs with dataclasses and simple-parsing

### Training Loop
- **[ ]** gradient checkpointing

### Distributed Backend
- **[done]** distributed backend file for:
  - **[done]**  single-gpu 
  - **[done]** DDP
  - **[done]** FSDP support
    - **[done]** test FSDP
    - **[ ]** use meta-device for large models 

### Matmuls Precision
- **[done]** integrate Quartet-II
  - **[ ]** test Quartet-II

### Dataset
- **[done]** sharded data reader
- **[ ]** sequence packing to move away from `llm-baselines` design (build a new dataset)
  - **[ ]** how to integrate this with RoPE in attention?

### Models
- **[done]** GPT model and model factory
- **[done]** Llama2/3
- **[done]** Qwen
- **[ ]** OLMo
- **[done]** custom models with different number of params for each family
- **[done]** add option to use custom attention implementation or FlashAttention from PyTorch. See comparison in [LLM-from-scratch]
  (https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb)
- **[ ]** implement KV-cache for inference
- **[done]** allocate only one mask (for attention) and sin/cos (for RoPE) in the model class and not for each attention module 

### Optimizer
- **[done]** optimizers package and factory
- **[ ]** learning rate scheduling
  - **[done]** cosine
  - **[ ]** linear
  - **[ ]** trapezoidal
  - **[ ]** interpret warmup as percentage of total steps if in [0, 1] and absolute number of steps if between (1, +oo)
  - **[ ]** add option to perform gradient clipping only when on the lr decay period (no clipping during warmup) - from 2402.00838 
    (OLMo-v1) section 3.2-Optimizer
- **[done]** integrate `MicroAdam`

### Managers
- **[done]** file with constants
- **[done]** logging
  - **[done]** wandb
  - **[done]** `tqdm` progress bars 
- **[done]** timing
  - **[ ]** add a `permanent_metric` dictionary that does not clean at each step to store the total running time of the process
- **[ ]** model checkpointing

### Inference
- **[ ]** KV-cache: https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms

### Evaluation
- **[done]** 

### Main Scripts
- **[done]** Pretraining

### Readability
- **[ ]** implement the factory methods inside the corresponding dataclass to follow OLMo-core standard

### Additional Readings
- **[ ]** skip optimizer
- **[ ]** z-loss
- **[ ]** @property decorator with fn_name.setter