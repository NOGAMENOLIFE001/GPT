"""
Text Generation Engine

High-level interface for generating text from a trained model.
"""

import torch
from typing import Generator

from model.model import LanguageModel
from tokenizer.bpe import BPETokenizer
from inference.decoding import DecodingConfig, sample_next_token


class GenerationEngine:
    """
    Text generation engine with streaming support.
    """

    def __init__(
        self,
        model: LanguageModel,
        tokenizer: BPETokenizer,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: DecodingConfig | None = None,
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input text prompt
            config: Decoding configuration

        Returns:
            Generated text (including prompt)
        """
        if config is None:
            config = DecodingConfig()

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)  # [1, seq]

        generated = input_ids[0].tolist()

        for _ in range(config.max_new_tokens):
            # Forward pass
            logits = self.model(input_ids)  # [1, seq, vocab]
            next_logits = logits[0, -1, :]  # [vocab]

            # Sample next token
            next_token = config.decode_next(next_logits, generated)
            next_token_id = next_token.item()

            # Check stop tokens
            if next_token_id in config.stop_tokens:
                break

            # Append to generated
            generated.append(next_token_id)

            # Update input for next iteration
            input_ids = torch.tensor([generated], dtype=torch.long, device=self.device)

            # Check context length
            if len(generated) >= self.model.context_length:
                break

        # Decode full sequence
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        config: DecodingConfig | None = None,
    ) -> Generator[str, None, None]:
        """
        Stream generated text token by token.

        Yields:
            Decoded text chunks as they are generated.
        """
        if config is None:
            config = DecodingConfig()

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        generated = input_ids[0].tolist()

        # First yield the prompt
        yield self.tokenizer.decode(generated, skip_special_tokens=True)

        for _ in range(config.max_new_tokens):
            logits = self.model(input_ids)
            next_logits = logits[0, -1, :]

            next_token = config.decode_next(next_logits, generated)
            next_token_id = next_token.item()

            if next_token_id in config.stop_tokens:
                break

            generated.append(next_token_id)
            input_ids = torch.tensor([generated], dtype=torch.long, device=self.device)

            # Yield only the new token's decoded text
            new_text = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
            yield new_text

            if len(generated) >= self.model.context_length:
                break

    @torch.no_grad()
    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get logits for input sequence."""
        return self.model(input_ids)


def load_model_and_tokenizer(
    checkpoint_path: str | None = None,
    model_variant: str = "tiny",
    dataset_name: str = "d1",
    device: str = "cpu",
) -> tuple[LanguageModel, BPETokenizer]:
    """Load model and tokenizer for inference."""
    from core.config import load_model_config, load_dataset_config
    from training.checkpoint import load_checkpoint

    # Load tokenizer
    dataset_config = load_dataset_config(dataset_name)
    tokenizer_path = Path(dataset_config["dataset_path"]) / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        # Train if not exists
        from tokenizer.bpe import train_tokenizer_from_config
        tokenizer = train_tokenizer_from_config(dataset_name)

    # Load model config
    model_config = load_model_config(model_variant)
    model = LanguageModel.from_config(model_config)

    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint = load_checkpoint(checkpoint_path, model, device=device)
        print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")

    model.to(device)
    model.eval()

    return model, tokenizer


from pathlib import Path

if __name__ == "__main__":
    # Quick test
    from core.config import load_model_config
    model_config = load_model_config("tiny")
    model = LanguageModel.from_config(model_config)

    # Dummy tokenizer
    tokenizer = BPETokenizer(vocab_size=model_config["vocab_size"])
    tokenizer.train(["hello world", "test text"])

    engine = GenerationEngine(model, tokenizer)

    # Test generation
    result = engine.generate("hello", DecodingConfig(max_new_tokens=10, strategy="greedy"))
    print(f"Generated: {result}")