"""
General Language Corpus Builder for Clyffe v0.2 (ARC-LB-01).

Fetches verified human-written corpora from Hugging Face datasets-server:
- wikimedia/wikipedia (20231101.simple, CC BY-SA 3.0) -> causal LM pretraining
- databricks/databricks-dolly-15k (CC BY-SA 3.0) -> general instructions, QA, summarization
- allenai/ai2_arc (ARC-Easy, CC BY-SA 4.0) -> human-written science QA
- openai/gsm8k (MIT) -> human-written math word problems
Plus deterministic project-generated multi-turn conversations, unit conversions,
arithmetic, identity, and out-of-scope refusal examples.
"""

import hashlib
import json
import os
import random
import urllib.request
from typing import Dict, List, Set


OOS_RESPONSE = "I don't have sufficient trained knowledge to answer this question."
OUT_DIR = "datasets/processed/general_v02"


def fetch_hf_rows(dataset: str, config: str, split: str, total: int, page_size: int = 100) -> List[dict]:
    rows: List[dict] = []
    offset = 0
    while offset < total:
        length = min(page_size, total - offset)
        url = (
            f"https://datasets-server.huggingface.co/rows?"
            f"dataset={urllib.parse.quote(dataset, safe='')}"
            f"&config={urllib.parse.quote(config, safe='')}"
            f"&split={urllib.parse.quote(split, safe='')}"
            f"&offset={offset}&length={length}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "ARC-III-Clyffe/0.2"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        batch = [r["row"] for r in data.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
    return rows


def build_synthetic_general_tasks(rng: random.Random):
    """Deterministic arithmetic, unit conversions, multi-turn context, and OOS refusal."""
    sft_rows: List[dict] = []

    # 1. Arithmetic & Unit Conversions (train 1..200, holdout 201..999)
    train_math, heldout_math = [], []
    for _ in range(450):
        a, b = rng.randint(10, 200), rng.randint(10, 200)
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            train_math.append({"turns": [(f"Calculate {a} + {b}.", f"{a} + {b} = {a + b}.")], "category": "math"})
        elif op == "-":
            hi, lo = max(a, b), min(a, b)
            train_math.append({"turns": [(f"Calculate {hi} - {lo}.", f"{hi} - {lo} = {hi - lo}.")], "category": "math"})
        else:
            x, y = rng.randint(2, 25), rng.randint(2, 25)
            train_math.append({"turns": [(f"What is {x} * {y}?", f"{x} * {y} = {x * y}.")], "category": "math"})

    for _ in range(150):
        km = rng.randint(1, 50)
        train_math.append({"turns": [(f"Convert {km} kilometres into metres.", f"{km} kilometres is {km * 1000} metres.")], "category": "conversion"})
        m = rng.randint(1, 20) * 100
        train_math.append({"turns": [(f"Convert {m} metres into centimetres.", f"{m} metres is {m * 100} centimetres.")], "category": "conversion"})

    for _ in range(100):
        a, b = rng.randint(201, 700), rng.randint(201, 700)
        heldout_math.append({"turns": [(f"Calculate {a} + {b}.", f"{a} + {b} = {a + b}.")], "category": "heldout_math"})

    # 2. Multi-turn contextual conversations
    names = ["Hari", "Priya", "Alex", "Maya", "Rohan", "Asha", "Liam", "Neha", "Kiran", "Sara"]
    topics = [
        ("Explain gravity in simple English.", "Gravity is the natural force that pulls objects with mass toward each other, keeping planets in orbit and keeping us on Earth."),
        ("What is the difference between RAM and ROM?", "RAM is temporary, fast memory used for active programs and cleared when power is off, while ROM is permanent memory storing essential startup instructions."),
        ("Why does ice melt?", "Ice melts when it absorbs heat energy, causing its solid water molecules to vibrate faster and break free into liquid water at 0 degrees Celsius."),
        ("What happens when water boils?", "When water boils at 100 degrees Celsius, heat turns liquid water into water vapor (steam) bubbles that rise to the surface."),
        ("Explain how electricity works.", "Electricity is the flow of electric charge, usually electrons moving through a conductor such as copper wire when driven by a voltage."),
        ("Explain photosynthesis.", "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to create glucose for energy and release oxygen."),
    ]
    for nm in names:
        q, a = rng.choice(topics)
        sft_rows.append({
            "turns": [
                (f"My name is {nm}.", f"Nice to meet you, {nm}!"),
                ("What is my name?", f"Your name is {nm}."),
                (q, a),
            ],
            "category": "multi_turn",
        })
        sft_rows.append({
            "turns": [
                ("Hello", "Hello! I am Clyffe, developed under ARC III. How can I help you?"),
                ("Who are you?", "I am Clyffe, a native LittleBit-2 language model developed under ARC III."),
            ],
            "category": "identity",
        })

    for q, a in topics:
        sft_rows.append({"turns": [(q, a)], "category": "science_explain"})

    # 3. Out-of-scope refusal
    oos_prompts = [
        "What is the stock price of Apple today?",
        "Predict tomorrow's lottery numbers.",
        "What is the weather in Tokyo right now?",
        "Who won yesterday's football match?",
        "Give me a medical diagnosis for chest pain.",
        "Write a legal contract for a commercial lease.",
        "What is the current price of Bitcoin?",
        "Who is the current prime minister of Freedonia?",
    ]
    for p in oos_prompts:
        sft_rows.append({"turns": [(p, OOS_RESPONSE)], "category": "out_of_scope"})

    return sft_rows + train_math, heldout_math


def main():
    import urllib.parse  # ensure available inside fetch_hf_rows
    globals()["urllib"] = urllib

    rng = random.Random(2026)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Fetching Simple Wikipedia (20231101.simple)...")
    wiki_rows = fetch_hf_rows("wikimedia/wikipedia", "20231101.simple", "train", total=1200, page_size=100)
    print(f"Fetched {len(wiki_rows)} Simple Wikipedia articles.")

    print("Fetching Databricks Dolly-15k...")
    dolly_rows = fetch_hf_rows("databricks/databricks-dolly-15k", "default", "train", total=1500, page_size=100)
    print(f"Fetched {len(dolly_rows)} Dolly-15k instructions.")

    print("Fetching AI2 ARC-Easy...")
    arc_rows = fetch_hf_rows("allenai/ai2_arc", "ARC-Easy", "train", total=600, page_size=100)
    print(f"Fetched {len(arc_rows)} ARC-Easy questions.")

    print("Fetching GSM8K...")
    gsm_rows = fetch_hf_rows("openai/gsm8k", "main", "train", total=400, page_size=100)
    print(f"Fetched {len(gsm_rows)} GSM8K problems.")

    # Pretraining texts (deduplicated)
    seen_texts: Set[str] = set()
    pretrain_docs: List[dict] = []
    for w in wiki_rows:
        txt = (w.get("title", "") + "\n" + w.get("text", "")).strip()
        if len(txt) < 120:
            continue
        # Keep first 1200 chars per article for balanced stage-1 diversity
        txt = txt[:1200].strip()
        h = hashlib.sha256(txt.encode("utf-8")).hexdigest()
        if h not in seen_texts:
            seen_texts.add(h)
            pretrain_docs.append({"text": txt, "source": "wikimedia/wikipedia:simple"})

    rng.shuffle(pretrain_docs)
    n_val_pt = max(50, int(0.08 * len(pretrain_docs)))
    pretrain_val = pretrain_docs[:n_val_pt]
    pretrain_train = pretrain_docs[n_val_pt:]

    # SFT records
    sft_all: List[dict] = []
    seen_sft: Set[str] = set()

    for d in dolly_rows:
        inst = (d.get("instruction") or "").strip()
        ctx = (d.get("context") or "").strip()
        resp = (d.get("response") or "").strip()
        if not inst or not resp or len(resp) > 600:
            continue
        prompt = f"{inst}\nContext: {ctx[:500]}" if ctx else inst
        k = prompt.lower()
        if k not in seen_sft:
            seen_sft.add(k)
            sft_all.append({"turns": [(prompt, resp)], "category": f"dolly_{d.get('category', 'general')}"})

    for a in arc_rows:
        q = (a.get("question") or "").strip()
        choices = a.get("choices") or {}
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        key = a.get("answerKey", "")
        if q and key in labels:
            ans = texts[labels.index(key)]
            if q.lower() not in seen_sft:
                seen_sft.add(q.lower())
                sft_all.append({"turns": [(q, ans)], "category": "arc_science"})

    for g in gsm_rows:
        q = (g.get("question") or "").strip()
        ans = (g.get("answer") or "").strip()
        if q and ans and len(ans) < 400 and q.lower() not in seen_sft:
            seen_sft.add(q.lower())
            sft_all.append({"turns": [(q, ans)], "category": "gsm8k"})

    synth_sft, heldout_math = build_synthetic_general_tasks(rng)
    for row in synth_sft:
        k = row["turns"][0][0].lower()
        if k not in seen_sft:
            seen_sft.add(k)
            sft_all.append(row)

    rng.shuffle(sft_all)
    n_sft_val = max(80, int(0.08 * len(sft_all)))
    n_sft_test = max(80, int(0.08 * len(sft_all)))
    sft_val = sft_all[:n_sft_val]
    sft_test = sft_all[n_sft_val : n_sft_val + n_sft_test] + heldout_math
    sft_train = sft_all[n_sft_val + n_sft_test :]

    files = {
        "pretrain_train.jsonl": pretrain_train,
        "pretrain_val.jsonl": pretrain_val,
        "sft_train.jsonl": sft_train,
        "sft_val.jsonl": sft_val,
        "sft_heldout_test.jsonl": sft_test,
    }
    manifest = {"dataset": "general_v02", "files": {}}
    for fname, rows in files.items():
        fpath = os.path.join(OUT_DIR, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        h = hashlib.sha256(open(fpath, "rb").read()).hexdigest()
        manifest["files"][fname] = {"count": len(rows), "sha256": h}

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

