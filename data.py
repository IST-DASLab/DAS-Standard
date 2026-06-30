"""
    Inspired from src/data/utils.py from llm-baselines
"""

import os
import numpy as np
import torch
from argparse import Namespace as ArgParseNamespace
import torch.distributed as dist


def get_dataset(cfg: ArgParseNamespace):
    root = cfg.data.source_dir
    if cfg.data.model_name == "c4":
        train_path = os.path.join(root, "c4", "train.bin")
        val_path = os.path.join(root, "c4", "val.bin")
        return train_path, val_path
    raise NotImplementedError(f'Unknown dataset {cfg.data.model_name}')


def get_data_readers(cfg, verbose=True):
    train_path, val_path = get_dataset(cfg)

    train_reader = SimpleDataReader(
        path=train_path,
        batch_size=cfg.data.batch_size,
        seq_len=cfg.model.seq_len,
        seed=cfg.data.seed,
        auto_shard=True)

    val_reader = SimpleDataReader(
        path=train_path,
        batch_size=cfg.data.batch_size,
        seq_len=cfg.model.seq_len,
        seed=cfg.data.seed,
        auto_shard=False)

    if verbose:
        print(f"Num training tokens: {train_reader.num_toks}")
        print(f"Num validation tokens: {val_reader.num_toks}")

    return train_reader, val_reader


class SimpleDataReader:
    def __init__(self, path, batch_size, seq_len, seed, auto_shard):
        self.path = path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.seed = seed
        self.auto_shard = auto_shard

        if auto_shard and dist.is_initialized(): # used for training data
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else: # used for validation data
            self.world_size = 1
            self.rank = 0

        self.data = None
        self.perm = None

        # total number of tokens in the dataset: 197_980_429_805 for C4
        self.num_toks = len(self._get_data())

        # how many sequences of shape "seq_len" we have in the dataset
        self.num_seqs = self._get_len() // self.seq_len

        # how many batches of size "batch_size" we obtain if we pack the sequences
        self.num_batches = (self.num_seqs - 1) // self.batch_size  # used to generate new perm
        self.epoch_token_offset = None

        self.last_epoch = None
        self.step = None

        self.reset()

    def reset(self):
        """
        This function should be called on the validation reader after each evaluation
        during training to make sure we always evaluate on the same first batch of data.
        """
        self.step = 0
        self.last_epoch = None
        self._shuffle_epoch(0)

    def _get_len(self):
        # Length in valid start indices for a sequence
        # Extra -1 to have a valid next token for the final token of the last idx
        return self.num_toks - self.seq_len - 1

    def _get_data(self):
        if self.data is None:
            return np.memmap(self.path, dtype=np.uint16, mode="r")
        return self.data

    def _shuffle_epoch(self, epoch):
        seed = self.seed + epoch
        rng = np.random.default_rng(seed)

        # we sample sequences of "seq_len" tokens at a time for one batch
        self.perm = rng.permutation(self.num_seqs - 1)

        # generate a new token offset
        self.epoch_token_offset = rng.integers(self.seq_len)

        # update epoch
        self.last_epoch = epoch

    def sample_batch(self, device='cuda'):
        data = self._get_data()

        # sample without replacement
        batch_idx = self.world_size * self.step + self.rank  # at each step we read "world_size" batches and index into it using "rank"

        epoch = batch_idx // self.num_batches
        if epoch != self.last_epoch:
            self._shuffle_epoch(epoch)

        epoch_idx = batch_idx % self.num_batches

        start = epoch_idx * self.batch_size
        end = start + self.batch_size
        idxs = self.perm[start:end] * self.seq_len + self.epoch_token_offset

        self.step += 1

        xy = np.stack([
            data[i: i + self.seq_len + 1]
            for i in idxs
        ]).astype(np.int64)

        x = torch.from_numpy(xy[:, :-1]).contiguous()
        y = torch.from_numpy(xy[:, 1:]).contiguous()

        if device == 'cuda':
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)

        return x, y