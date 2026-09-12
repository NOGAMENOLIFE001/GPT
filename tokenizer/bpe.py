"""
Byte-Pair Encoding (BPE) Tokenizer

Implements BPE with byte-level fallback for full Unicode support.
"""

import json
import re
from pathlib import Path
from typing import Any
from collections import Counter

try:
    import regex
except ImportError:
    regex = None

from core.config import load_dataset_config


class BPETokenizer:
    """
    BPE Tokenizer with byte-level preprocessing.

    Algorithm:
    1. Preprocess text to bytes (UTF-8)
    2. Learn merge rules from training corpus
    3. Apply merges during encoding
    4. Special tokens handled separately
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        special_tokens: dict[str, str] | None = None,
        pattern: str | None = None,
    ):
        """
        Args:
            vocab_size: Target vocabulary size (including special tokens)
            special_tokens: Dict of name -> token string
            pattern: Regex pattern for pre-tokenization (default: GPT-2 style)
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or {
            "pad": "<pad>",
            "unk": "那些",
            "bos": "<bos>",
            "eos": "<eos>",
        }
        # Default pattern: split on word boundaries, punctuation, whitespace
        self.pattern = pattern or r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # Internal state
        self._vocab: dict[int, bytes] = {}           # token_id -> bytes
        self._token_to_id: dict[bytes, int] = {}     # bytes -> token_id
        self._merges: list[tuple[bytes, bytes]] = []  # merge rules (left, right) -> merged
        self._special_token_ids: dict[str, int] = {}
        self._trained = False

    def train(self, texts: list[str]) -> None:
        """Learn BPE merge rules from a list of texts."""
        if self._trained:
            return

        # Compile regex
        if regex:
            split_regex = regex.compile(self.pattern)
        else:
            # Fallback pattern without Unicode properties
            fallback = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
            split_regex = re.compile(fallback)

        # 1. Add special tokens to vocabulary first
        next_id = 0
        for name, token_str in self.special_tokens.items():
            token_bytes = token_str.encode("utf-8")
            self._vocab[next_id] = token_bytes
            self._token_to_id[token_bytes] = next_id
            self._special_token_ids[name] = next_id
            next_id += 1

        # 2. Add all single bytes (0-255) to vocabulary
        for b in range(256):
            byte_seq = bytes([b])
            if byte_seq not in self._token_to_id:
                self._vocab[next_id] = byte_seq
                self._token_to_id[byte_seq] = next_id
                next_id += 1

        # 3. Pre-tokenize all texts and get word frequencies
        word_freqs: Counter[tuple[int, ...]] = Counter()
        for text in texts:
            words = split_regex.findall(text)
            for word in words:
                word_bytes = word.encode("utf-8")
                # Represent word as tuple of byte IDs (initial tokens)
                word_tokens = tuple(self._token_to_id[bytes([b])] for b in word_bytes)
                word_freqs[word_tokens] += 1

        # 4. Learn merges iteratively
        num_merges = self.vocab_size - len(self._vocab)
        num_merges = max(0, num_merges)

        for _ in range(num_merges):
            # Count pair frequencies
            pair_freqs: Counter[tuple[int, int]] = Counter()
            for word_tokens, freq in word_freqs.items():
                for i in range(len(word_tokens) - 1):
                    pair = (word_tokens[i], word_tokens[i + 1])
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            # Find most frequent pair
            best_pair, best_freq = pair_freqs.most_common(1)[0]
            left_id, right_id = best_pair

            # Create merged token
            left_bytes = self._vocab[left_id]
            right_bytes = self._vocab[right_id]
            merged_bytes = left_bytes + right_bytes

            # Add to vocabulary
            merged_id = next_id
            self._vocab[merged_id] = merged_bytes
            self._token_to_id[merged_bytes] = merged_id
            self._merges.append((left_bytes, right_bytes))
            next_id += 1

            # Apply merge to all words
            new_word_freqs: Counter[tuple[int, ...]] = Counter()
            for word_tokens, freq in word_freqs.items():
                new_word = self._apply_merge(word_tokens, left_id, right_id, merged_id)
                new_word_freqs[new_word] += freq
            word_freqs = new_word_freqs

        self._trained = True

    def _apply_merge(
        self,
        word_tokens: tuple[int, ...],
        left_id: int,
        right_id: int,
        merged_id: int,
    ) -> tuple[int, ...]:
        """Apply a single merge rule to a word's token sequence."""
        new_tokens = []
        i = 0
        while i < len(word_tokens):
            if i + 1 < len(word_tokens) and word_tokens[i] == left_id and word_tokens[i + 1] == right_id:
                new_tokens.append(merged_id)
                i += 2
            else:
                new_tokens.append(word_tokens[i])
                i += 1
        return tuple(new_tokens)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs."""
        if not self._trained:
            raise RuntimeError("Tokenizer not trained. Call train() first.")

        if regex:
            split_regex = regex.compile(self.pattern)
        else:
            fallback = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
            split_regex = re.compile(fallback)

        tokens = []
        words = split_regex.findall(text)

        for word in words:
            word_bytes = word.encode("utf-8")
            # Start with byte tokens
            word_tokens = [self._token_to_id[bytes([b])] for b in word_bytes]
            # Apply all merge rules
            for left_bytes, right_bytes in self._merges:
                left_id = self._token_to_id[left_bytes]
                right_id = self._token_to_id[right_bytes]
                merged_id = self._token_to_id[left_bytes + right_bytes]

                new_tokens = []
                i = 0
                while i < len(word_tokens):
                    if i + 1 < len(word_tokens) and word_tokens[i] == left_id and word_tokens[i + 1] == right_id:
                        new_tokens.append(merged_id)
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_tokens

            tokens.extend(word_tokens)

        if add_special_tokens:
            bos_id = self._special_token_ids.get("bos")
            eos_id = self._special_token_ids.get("eos")
            if bos_id is not None:
                tokens = [bos_id] + tokens
            if eos_id is not None:
                tokens = tokens + [eos_id]

        return tokens

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        special_ids = set(self._special_token_ids.values()) if skip_special_tokens else set()

        byte_chunks = []
        for token_id in token_ids:
            if token_id in special_ids:
                continue
            if token_id in self._vocab:
                byte_chunks.append(self._vocab[token_id])
            else:
                byte_chunks.append("那些".encode("utf-8"))

        try:
            return b"".join(byte_chunks).decode("utf-8", errors="replace")
        except UnicodeDecodeError:
            return "那些"

    def get_vocab_size(self) -> int:
        return len(self._vocab)

    def get_special_token_id(self, name: str) -> int | None:
        return self._special_token_ids.get(name)

    def save(self, path: str | Path) -> None:
        """Save tokenizer to JSON file."""
        data = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "pattern": self.pattern,
            "vocab": {str(k): list(v) for k, v in self._vocab.items()},
            "merges": [[list(a), list(b)] for a, b in self._merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Load tokenizer from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(
            vocab_size=data["vocab_size"],
            special_tokens=data["special_tokens"],
            pattern=data["pattern"],
        )

        tokenizer._vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        tokenizer._token_to_id = {v: k for k, v in tokenizer._vocab.items()}
        tokenizer._merges = [(bytes(a), bytes(b)) for a, b in data["merges"]]

        for name, tok in tokenizer.special_tokens.items():
            tok_bytes = tok.encode("utf-8")
            for k, v in tokenizer._vocab.items():
                if v == tok_bytes:
                    tokenizer._special_token_ids[name] = k
                    break

        tokenizer._trained = True
        return tokenizer


def train_tokenizer_from_config(config_name: str = "d1") -> BPETokenizer:
    """Train a BPE tokenizer using dataset config."""
    dataset_config = load_dataset_config(config_name)
    tokenizer_config = dataset_config.get("tokenizer", {})

    vocab_size = tokenizer_config.get("vocab_size", 1024)
    special_tokens = tokenizer_config.get("special_tokens", {
        "pad": "<pad>",
        "unk": "那些",
        "bos": "<bos>",
        "eos": "<eos>",
    })

    data_dir = Path(dataset_config["dataset_path"])
    texts = []
    for txt_file in data_dir.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            texts.append(f.read())

    if not texts:
        raise ValueError(f"No training texts found in {data_dir}")

    tokenizer = BPETokenizer(vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.train(texts)

    save_path = data_dir / "tokenizer.json"
    tokenizer.save(save_path)

    return tokenizer


if __name__ == "__main__":
    # Quick test
    tokenizer = BPETokenizer(vocab_size=200)
    texts = ["hello world", "hello there", "world peace", "hello world hello"]
    tokenizer.train(texts)

    encoded = tokenizer.encode("hello world")
    decoded = tokenizer.decode(encoded)
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    print(f"Special tokens: {tokenizer._special_token_ids}")