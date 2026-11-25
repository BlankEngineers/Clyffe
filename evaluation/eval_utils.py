"""
Shared evaluation utilities for ARC-LB-01 capability & boundary testing.
"""

import json
import re
from typing import Callable, Dict, List

from src.inference.runtime import ARCLB01Runtime


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def token_f1(pred: str, ref: str) -> float:
    p_toks = normalize_text(pred).split()
    r_toks = normalize_text(ref).split()
    if not p_toks or not r_toks:
        return 1.0 if p_toks == r_toks else 0.0
    common = {}
    for t in p_toks:
        common[t] = min(p_toks.count(t), r_toks.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(p_toks)
    rec = num_same / len(r_toks)
    return 2 * prec * rec / (prec + rec)


def load_jsonl_category(paths: List[str], category: str, limit: int = 100) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("category") == category:
                    rows.append(rec)
    return rows[:limit]


def evaluate_category(
    runtime: ARCLB01Runtime,
    rows: List[Dict[str, str]],
    match_fn: Callable[[str, str], bool] = None,
    max_new_tokens: int = 40,
) -> Dict[str, object]:
    if match_fn is None:
        match_fn = lambda pred, ref: token_f1(pred, ref) >= 0.5

    exact = 0
    semantic = 0
    samples = []
    for i, r in enumerate(rows):
        pred = runtime.respond(r["input"], max_new_tokens=max_new_tokens)
        is_exact = normalize_text(pred) == normalize_text(r["output"])
        is_sem = bool(match_fn(pred, r["output"]))
        exact += int(is_exact)
        semantic += int(is_sem)
        if i < 5:
            samples.append({"input": r["input"], "expected": r["output"], "predicted": pred, "pass": is_sem})

    n = max(1, len(rows))
    return {
        "count": len(rows),
        "exact_accuracy": round(exact / n, 4),
        "semantic_accuracy": round(semantic / n, 4),
        "samples": samples,
    }

