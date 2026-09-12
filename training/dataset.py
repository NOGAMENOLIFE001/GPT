"""
Dataset and DataLoader for Language Modeling

Handles tokenization, batching, and train/val splits.
"""

import random
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer.bpe import BPETokenizer, train_tokenizer_from_config
from core.config import load_dataset_config, load_model_config


class TextDataset(Dataset):
    """
    Dataset for causal language modeling.

    Loads text files, tokenizes them, and creates fixed-length sequences.
    """

    def __init__(
        self,
        tokenizer: BPETokenizer,
        texts: list[str],
        max_seq_len: int,
        stride: int | None = None,
    ):
        """
        Args:
            tokenizer: Trained BPE tokenizer
            texts: List of text strings
            max_seq_len: Maximum sequence length (context window)
            stride: Stride for sliding window (default: max_seq_len // 2)
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride or (max_seq_len // 2)

        # Tokenize all texts
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            all_tokens.extend(tokens)

        self.all_tokens = all_tokens
        self.num_sequences = max(0, (len(all_tokens) - max_seq_len) // self.stride + 1)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.stride
        end = start + self.max_seq_len

        # Input: tokens[start:end]
        # Target: tokens[start+1:end+1] (shifted by 1 for next-token prediction)
        input_ids = torch.tensor(self.all_tokens[start:end], dtype=torch.long)
        target_ids = torch.tensor(self.all_tokens[start + 1:end + 1], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
        }


def load_texts_from_dir(data_dir: Path) -> list[str]:
    """Load all .txt files from a directory."""
    texts = []
    for txt_file in data_dir.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts


def create_datasets_from_config(
    config_name: str = "d1",
    model_config_name: str = "tiny",
) -> tuple[TextDataset, TextDataset, BPETokenizer]:
    """
    Create train and validation datasets from config.

    Returns:
        train_dataset, val_dataset, tokenizer
    """
    dataset_config = load_dataset_config(config_name)
    model_config = load_model_config(model_config_name)

    data_dir = Path(dataset_config["dataset_path"])
    max_seq_len = model_config["context_length"]
    train_split = dataset_config.get("train_split", 0.9)

    # Load texts
    texts = load_texts_from_dir(data_dir)
    if not texts:
        raise ValueError(f"No texts found in {data_dir}")

    # Train tokenizer on ALL texts first
    tokenizer = train_tokenizer_from_config(config_name)

    # Encode all texts into a single token stream
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        all_tokens.extend(tokens)

    # Split tokens into train/val
    split_idx = int(len(all_tokens) * train_split)
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]

    # Create datasets from token lists
    class TokenListDataset(Dataset):
        def __init__(self, tokens: list[int], max_seq_len: int, stride: int | None = None):
            self.tokens = tokens
            self.max_seq_len = max_seq_len
            self.stride = stride or (max_seq_len // 2)
            self.num_sequences = max(0, (len(tokens) - max_seq_len) // self.stride + 1)

        def __len__(self) -> int:
            return self.num_sequences

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            start = idx * self.stride
            end = start + self.max_seq_len
            input_ids = torch.tensor(self.tokens[start:end], dtype=torch.long)
            target_ids = torch.tensor(self.tokens[start + 1:end + 1], dtype=torch.long)
            return {"input_ids": input_ids, "target_ids": target_ids}

    train_dataset = TokenListDataset(train_tokens, max_seq_len)
    val_dataset = TokenListDataset(val_tokens, max_seq_len)

    return train_dataset, val_dataset, tokenizer


def create_dataloaders(
    train_dataset: TextDataset,
    val_dataset: TextDataset,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


class InfiniteDataLoader:
    """Wrapper that cycles through a dataloader infinitely."""

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self._iterator = iter(dataloader)

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> dict:
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataloader)
            return next(self._iterator)


if __name__ == "__main__":
    # Quick test
    train_ds, val_ds, tokenizer = create_datasets_from_config("d1", "tiny")
    print(f"Train sequences: {len(train_ds)}")
    print(f"Val sequences: {len(val_ds)}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    sample = train_ds[0]
    print(f"Sample input shape: {sample['input_ids'].shape}")
    print(f"Sample target shape: {sample['target_ids'].shape}")
    print(f"Decoded: {tokenizer.decode(sample['input_ids'].tolist())}")