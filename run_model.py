"""
Command-Line Interface for ARC-LB-01 Native Offline Inference.
"""

import argparse
from src.inference.runtime import ARCLB01Runtime


def main():
    parser = argparse.ArgumentParser(description="Run ARC-LB-01 offline")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/ARC-LB-01-FINAL_best.pt",
    )
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()

    runtime = ARCLB01Runtime(
        checkpoint_path=args.checkpoint,
        tokenizer_dir=args.tokenizer,
        device=args.device,
    )

    if args.prompt is not None:
        print(f"ARC-LB-01: {runtime.respond(args.prompt)}")
        return

    print("Tesla (Blank Engineers | ARC III Ecosystem) Offline Runtime (type 'exit' or 'quit' to stop)")
    print(f"Loaded: {runtime.metadata['checkpoint_identifier']} | {runtime.metadata['weight_representation']}")
    while True:
        try:
            user_in = input("User: ").strip()
        except EOFError:
            break
        if user_in.lower() in ("exit", "quit"):
            break
        print(f"ARC-LB-01: {runtime.respond(user_in)}")


if __name__ == "__main__":
    main()

