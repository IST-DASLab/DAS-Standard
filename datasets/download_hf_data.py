from huggingface_hub import list_repo_files, hf_hub_download
import os

repo_id = "ISTA-DASLab/C4-tokenized-llama2"
local_dir = "/tmp/DAS-Standard/datasets/"
os.makedirs(local_dir, exist_ok=True)

# List all files in the repo
files = list_repo_files(repo_id, repo_type="dataset")

# Filter for chunk files
chunk_files = [f for f in files if f.startswith("chunk_") or f == "val.bin"]

# Download all chunks
for f in chunk_files:
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=f,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print(f"Downloaded {f} -> {local_path}")

##### after download, run "cat chunk_* > train.bin" in linux

# Optional: concatenate into a single file
# import glob, shutil
#
# chunk_paths = sorted(glob.glob(os.path.join(local_dir, "chunk_*")))
# output_path = os.path.join(local_dir, "train.bin")
#
# with open(output_path, "wb") as out_f:
#     for cp in chunk_paths:
#         with open(cp, "rb") as in_f:
#             shutil.copyfileobj(in_f, out_f)
#
# print(f"All chunks concatenated into {output_path}")
