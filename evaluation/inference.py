"""
Native Inference Latency, Memory, and Offline Integrity Benchmark.
"""

import json
import time
import tracemalloc
from typing import Dict
import torch

from src.inference.runtime import ARCLB01Runtime


def evaluate_inference(ckpt_path: str, device: str = "cpu") -> Dict[str, object]:
    tracemalloc.start()
    t_load0 = time.perf_counter()
    runtime = ARCLB01Runtime(ckpt_path, device=device)
    load_ms = (time.perf_counter() - t_load0) * 1000.0

    prompts = ["Hello", "Who are you?", "What is gravity?", "25 + 35", "What is the stock price of Apple today?"]
    latencies = []
    outputs = {}
    for p in prompts:
        t0 = time.perf_counter()
        out = runtime.respond(p, max_new_tokens=32)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        outputs[p] = out

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "checkpoint": ckpt_path,
        "device": device,
        "load_time_ms": round(load_ms, 2),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        "peak_python_ram_mb": round(peak_bytes / (1024 * 1024), 2),
        "metadata": runtime.metadata,
        "sample_outputs": outputs,
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/ARC-LB-01-FINAL_best.pt"
    print(json.dumps(evaluate_inference(path), indent=2))

