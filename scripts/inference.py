"""
Inference Entry Point

Usage:
    python -m scripts.inference checkpoints/EXP-001/best.pt --prompt "Hello world"
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.engine import GenerationEngine, load_model_and_tokenizer
from inference.decoding import DecodingConfig


def main():
    parser = argparse.ArgumentParser(description="Generate text from trained model")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--prompt", default="", help="Input prompt")
    parser.add_argument("--model", default="tiny", help="Model variant")
    parser.add_argument("--dataset", default="d1", help="Dataset name (for tokenizer)")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--strategy", default="top_p", choices=["greedy", "top_k", "top_p", "temperature"])
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--stream", action="store_true", help="Stream output token by token")
    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        checkpoint_path=args.checkpoint,
        model_variant=args.model,
        dataset_name=args.dataset,
        device=args.device,
    )

    # Create generation engine
    engine = GenerationEngine(model, tokenizer, device=args.device)

    # Decoding config
    config = DecodingConfig(
        strategy=args.strategy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_tokens,
    )

    # Get prompt
    prompt = args.prompt
    if not prompt:
        prompt = input("Enter prompt: ")

    print(f"\nPrompt: {prompt}")
    print("-" * 50)

    # Generate
    if args.stream:
        for chunk in engine.generate_stream(prompt, config):
            print(chunk, end="", flush=True)
        print()
    else:
        result = engine.generate(prompt, config)
        print(result)


if __name__ == "__main__":
    main()