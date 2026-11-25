"""
Command-Line Interface for Clyffe (ARC-LB-01) Native Offline Inference.
"""

import argparse
from src.inference.runtime import ARCLB01Runtime


def main():
    parser = argparse.ArgumentParser(description="Run Clyffe offline")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/clyffe-v0.1.1-C-smoothsign-20261125_best.pt",
    )
    parser.add_argument("--tokenizer", type=str, default="checkpoints/clyffe_v0.1.1_tokenizer")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()

    runtime = ARCLB01Runtime(
        checkpoint_path=args.checkpoint,
        tokenizer_dir=args.tokenizer,
        device=args.device,
    )

    if args.prompt is not None:
        print(f"Clyffe: {runtime.respond(args.prompt)}")
        return

    print("Clyffe v0.1.1 (Blank Engineers | ARC III Ecosystem) Offline Runtime (type 'exit' or 'quit' to stop)")
    print(f"Loaded: {runtime.metadata['checkpoint_identifier']} | {runtime.metadata['weight_representation']}")
    while True:
        try:
            user_in = input("User: ").strip()
        except EOFError:
            break
        if user_in.lower() in ("exit", "quit"):
            break
        print(f"Clyffe: {runtime.respond(user_in)}")


if __name__ == "__main__":
    main()

