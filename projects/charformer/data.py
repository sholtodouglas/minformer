"""Tokenizes text and creates dataloader."""

import numpy as np
from typing import Dict, List, Tuple

class CharDataset:
    """
    Emits batches of characters
    """
    @staticmethod
    def get_default_config():
        class Config:
            block_size: int = 256
        return Config()

    def __init__(self, config, data: str):
        self.config = config
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print(f'data has {data_size} characters, {vocab_size} unique.')
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def get_block_size(self) -> int:
        return self.config.block_size

    def __len__(self) -> int:
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as NumPy arrays
        x = np.array(dix[:-1], dtype=np.int32)
        y = np.array(dix[1:], dtype=np.int32)
        return x, y
