### TODO List for DAS-Standard

---

### Python Environment
- **[done]** requirements file

### Command Line Arguments
- **[done]** configs with dataclasses and simple-parsing

### Distributed Backend
- **[2 / 3]** distributed backend file for:
  - **[done]**  single-gpu 
  - **[done]** DDP
  - **[ ]** FSDP support

### Matmuls Precision
- **[ ]** integrate Quartet-II

### Dataset
- **[done]** sharded data reader

### Models
- **[done]** GPT model and model factory
- **[ ]** OLMo
- **[ ]** Llama
- **[ ]** custom models with different number of params

### Optimizer
- **[done]** optimizers package and factory
- **[ ]** learning rate scheduling
  - **[done]** cosine
  - **[ ]** linear
  - **[ ]** trapezoidal
  - **[done]** renamed params related to lr scheduling

### Managers
- **[done]** file with constants
- **[done]** logging
  - **[done]** wandb
  - **[done]** `tqdm` progress bars 
- **[done]** timing
- **[ ]** model checkpointing

### Evaluation
- **[done]** 

### Main Scripts
- Pretraining