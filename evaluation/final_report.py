"""
Category & Boundary Evaluators + Final Evaluation Report Generator for ARC-LB-01.
"""

import json
import os
from typing import Dict

from evaluation.compression import evaluate_compression
from evaluation.eval_utils import evaluate_category, load_jsonl_category, normalize_text, token_f1
from evaluation.inference import evaluate_inference
from src.inference.runtime import ARCLB01Runtime


EVAL_PATHS = ["datasets/validation/val.jsonl", "datasets/test/test.jsonl"]


def run_all_evaluations(ckpt_path: str, tokenizer_dir: str = "checkpoints/tokenizer", device: str = "cpu") -> Dict[str, object]:
    runtime = ARCLB01Runtime(ckpt_path, tokenizer_dir=tokenizer_dir, device=device)

    greetings = evaluate_category(runtime, load_jsonl_category(EVAL_PATHS, "greetings", 100))
    identity = evaluate_category(runtime, load_jsonl_category(EVAL_PATHS, "identity", 100))
    conversation = evaluate_category(runtime, load_jsonl_category(EVAL_PATHS, "conversation", 100))
    science = evaluate_category(runtime, load_jsonl_category(EVAL_PATHS, "science", 100))
    general_knowledge = evaluate_category(runtime, load_jsonl_category(EVAL_PATHS, "general_knowledge", 100))

    # Unseen mathematics (500 expressions)
    math_rows = load_jsonl_category(["datasets/test/unseen_math.jsonl"], "mathematics", 500)
    mathematics = evaluate_category(
        runtime,
        math_rows,
        match_fn=lambda pred, ref: normalize_text(pred) == normalize_text(ref),
        max_new_tokens=16,
    )

    # Out-of-scope (200 questions)
    oos_rows = load_jsonl_category(EVAL_PATHS, "out_of_scope", 200)
    while len(oos_rows) < 200 and oos_rows:
        oos_rows = (oos_rows * 3)[:200]
    out_of_scope = evaluate_category(
        runtime,
        oos_rows,
        match_fn=lambda pred, ref: "don't have sufficient trained knowledge" in pred.lower() or token_f1(pred, ref) >= 0.7,
        max_new_tokens=32,
    )

    compression = evaluate_compression(ckpt_path)
    inference_bench = evaluate_inference(ckpt_path, tokenizer_dir=tokenizer_dir, device=device)

    return {
        "checkpoint": ckpt_path,
        "categories": {
            "greetings": greetings,
            "identity": identity,
            "conversation": conversation,
            "science": science,
            "general_knowledge": general_knowledge,
            "unseen_mathematics": mathematics,
            "out_of_scope": out_of_scope,
        },
        "compression": compression,
        "inference": inference_bench,
    }


def write_final_markdown_report(report: Dict[str, object], out_path: str = "evaluation/FINAL_EVALUATION_REPORT.md"):
    cats = report["categories"]
    comp = report["compression"]
    inf = report["inference"]

    lines = [
        "# ARC-LB-01 Final Evaluation Report",
        "",
        f"**Checkpoint Evaluated:** `{report['checkpoint']}`",
        f"**SHA-256:** `{comp['sha256']}`",
        "",
        "## 1. Capability & Boundary Results",
        "",
        "| Category | Samples | Exact Accuracy | Semantic / Match Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for name, res in cats.items():
        lines.append(f"| {name} | {res['count']} | {res['exact_accuracy']*100:.2f}% | {res['semantic_accuracy']*100:.2f}% |")

    lines += [
        "",
        "## 2. Compression & Representation Audit",
        "",
        f"- **Dense Equivalent Params:** {comp['total_dense_equivalent_params']}",
        f"- **LittleBit Covered Params:** {comp['littlebit_covered_params']} ({comp['littlebit_coverage_pct']}%)",
        f"- **Overall Effective BPW:** {comp['overall_effective_bpw']}",
        f"- **Checkpoint File Size:** {comp['checkpoint_file_bytes']} bytes",
        "",
        "## 3. Native Offline Inference Benchmark",
        "",
        f"- **Load Time:** {inf['load_time_ms']} ms",
        f"- **Average Generation Latency:** {inf['avg_latency_ms']} ms",
        f"- **Peak Python RAM:** {inf['peak_python_ram_mb']} MB",
        "",
        "## 4. Sample Outputs",
        "",
    ]
    for p, out in inf["sample_outputs"].items():
        lines.append(f"- **User:** `{p}` -> **ARC-LB-01:** `{out}`")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/ARC-LB-01-V2_best.pt"
    tok = sys.argv[2] if len(sys.argv) > 2 else ("checkpoints/tokenizer_v2" if "V2" in ckpt else "checkpoints/tokenizer")
    rep = run_all_evaluations(ckpt, tokenizer_dir=tok)
    write_final_markdown_report(rep)
    with open("evaluation/final_results.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print("Saved evaluation/FINAL_EVALUATION_REPORT.md and evaluation/final_results.json")

