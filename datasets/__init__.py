import os
from argparse import Namespace as ArgParseNamespace

from .data_reader import DataReader


def get_dataset(cfg: ArgParseNamespace):
    root = cfg.dataset.dataset_path
    if cfg.dataset.dataset_name == "c4":
        train_path = os.path.join(root, "c4", "train.bin")
        val_path = os.path.join(root, "c4", "val.bin")
        return train_path, val_path
    raise NotImplementedError(f'Unknown dataset {cfg.dataset.dataset_name}')


def get_data_readers(cfg):
    train_path, val_path = get_dataset(cfg)

    train_reader = DataReader(
        path=train_path,
        batch_size=cfg.training.local_batch_size,
        seq_len=cfg.model.seq_len,
        seed=cfg.training.seed,
        auto_shard=True)

    val_reader = DataReader(
        path=val_path,
        batch_size=cfg.training.local_batch_size,
        seq_len=cfg.model.seq_len,
        seed=cfg.training.seed,
        auto_shard=True)

    return train_reader, val_reader
