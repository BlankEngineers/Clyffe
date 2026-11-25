"""
Clyffe v0.1.2 General Language Corpus Builder (ARC-LB-01).

Combines verified human-written Hugging Face datasets (Simple Wikipedia,
Dolly-15k, ARC-Easy, GSM8K) with diverse natural/empathetic conversations,
science/tech explanations, and programmatically verified arithmetic/algebra/unit
conversions. Strictly excludes all Phase 11 held-out evaluation prompts.
"""

import hashlib
import json
import os
import random
import urllib.parse
import urllib.request
from typing import Dict, List, Set


OOS_RESPONSE = "I don't have sufficient trained knowledge to answer this question."
OUT_DIR = "datasets/processed/clyffe_v012"

# Exact Phase 7 & Phase 11 held-out prompts (NEVER included in training)
HELDOUT_PROMPTS = [
    # General Conversation
    ("Hello.", "conversation"),
    ("How are you?", "conversation"),
    ("I am feeling happy.", "conversation"),
    ("I am not in a good mood.", "conversation"),
    ("Can you help me?", "conversation"),
    ("Tell me something interesting.", "conversation"),
    # General Knowledge
    ("What is the solar system?", "general_knowledge"),
    ("Why does the moon shine?", "general_knowledge"),
    ("How does rain form?", "general_knowledge"),
    ("What is the purpose of the human heart?", "general_knowledge"),
    # Science
    ("Explain gravity.", "science"),
    ("What is an atom?", "science"),
    ("How does photosynthesis work?", "science"),
    ("Why does ice melt?", "science"),
    ("What happens when water boils?", "science"),
    # Technology
    ("What is a computer?", "technology"),
    ("Explain RAM and ROM.", "technology"),
    ("What is artificial intelligence?", "technology"),
    ("How does the internet work?", "technology"),
    # Mathematics (Phase 7 & 11)
    ("2*2", "mathematics"),
    ("12*3", "mathematics"),
    ("25+35", "mathematics"),
    ("345+678", "mathematics"),
    ("91-36", "mathematics"),
    ("225/15", "mathematics"),
    ("17*9", "mathematics"),
    # General Instructions
    ("Explain gravity in simple English.", "instructions"),
    ("Write a paragraph about computers.", "instructions"),
    ("Summarize a short passage: Plants convert sunlight, water, and carbon dioxide into food and oxygen through photosynthesis.", "instructions"),
    ("Compare two planets: Earth and Mars.", "instructions"),
    ("Explain the water cycle step by step.", "instructions"),
]
HELDOUT_SET = {p.lower().strip() for p, _ in HELDOUT_PROMPTS}


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
        req = urllib.request.Request(url, headers={"User-Agent": "ARC-III-Clyffe/0.1.2"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        batch = [r["row"] for r in data.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
    return rows


def build_curated_general_sft(rng: random.Random) -> List[dict]:
    rows: List[dict] = []

    # 1. Diverse empathetic & natural multi-turn conversations (different wording from held-out)
    mood_dialogues = [
        [("I am feeling sad today.", "I am sorry to hear that you are feeling sad. If you want to talk about what is bothering you, I am here to listen."), ("Thanks for listening.", "You are welcome. Take things one step at a time, and let me know if I can help.")],
        [("I feel really stressed out.", "Stress can be exhausting. Taking a short break or a few slow breaths can help. Would you like to share what is causing it?")],
        [("I am having a rough day.", "I am sorry your day has been rough. I am here if you want to vent or talk through something.")],
        [("I feel lonely right now.", "Feeling lonely is hard. Remember you are not alone here, and we can chat about anything you like.")],
        [("I am excited about my project!", "That is wonderful to hear! What are you working on?")],
        [("I feel tired and unmotivated.", "Rest is important when you feel drained. Maybe start with one small, easy task or take a short break.")],
        [("I am worried about my exam.", "It is natural to feel nervous before an exam. Focusing on one topic at a time can help build confidence.")],
        [("Good morning, how is it going?", "Good morning! Everything is running smoothly. How is your day starting?")],
        [("Hey there, can we chat?", "Of course! What would you like to talk about today?")],
        [("Thank you so much for your help.", "You are very welcome! Let me know whenever you need anything else.")],
    ]
    for turns in mood_dialogues:
        rows.append({"turns": turns, "category": "empathetic_chat"})

    # 2. Multi-turn name & context memory
    for nm in ["Hari", "Priya", "Alex", "Maya", "Rohan", "Neha", "Liam", "Asha", "Kiran", "Sara"]:
        rows.append({
            "turns": [
                (f"My name is {nm}.", f"Nice to meet you, {nm}! How can I help you today?"),
                ("What is my name?", f"Your name is {nm}."),
            ],
            "category": "multi_turn_memory",
        })

    # 3. Identity (Clyffe / ARC III)
    rows.extend([
        {"turns": [("Who are you?", "I am Clyffe, an experimental LittleBit-2 language model developed under ARC III.")], "category": "identity"},
        {"turns": [("What is your name?", "My name is Clyffe.")], "category": "identity"},
        {"turns": [("Who built you?", "I was built by Blank Engineers under the ARC III research ecosystem using LittleBit-2.")], "category": "identity"},
        {"turns": [("What is ARC-LB-01?", "ARC-LB-01 is the research code name behind Clyffe.")], "category": "identity"},
    ])

    # 4. Programmatic verified mathematics (spaced & unspaced, excluding Phase 7/11 heldout)
    reserved_math = {"2*2", "12*3", "25+35", "345+678", "91-36", "225/15", "17*9"}
    math_count = 0
    while math_count < 900:
        op = rng.choice(["+", "-", "*", "/", "pct", "cmp", "alg", "conv"])
        if op == "+":
            a, b = rng.randint(1, 500), rng.randint(1, 500)
            expr = rng.choice([f"{a}+{b}", f"{a} + {b}", f"Calculate {a} + {b}."])
            ans = str(a + b)
        elif op == "-":
            a = rng.randint(10, 500)
            b = rng.randint(1, a)
            expr = rng.choice([f"{a}-{b}", f"{a} - {b}", f"What is {a} - {b}?"])
            ans = str(a - b)
        elif op == "*":
            a, b = rng.randint(2, 25), rng.randint(2, 25)
            expr = rng.choice([f"{a}*{b}", f"{a} * {b}", f"Multiply {a} and {b}."])
            ans = str(a * b)
        elif op == "/":
            b = rng.randint(2, 20)
            q = rng.randint(2, 25)
            a = b * q
            expr = rng.choice([f"{a}/{b}", f"{a} / {b}", f"Divide {a} by {b}."])
            ans = str(q)
        elif op == "pct":
            p = rng.choice([10, 20, 25, 50, 75])
            n = rng.randint(1, 20) * 20
            expr = f"What is {p}% of {n}?"
            ans = str((p * n) // 100)
        elif op == "cmp":
            a, b = rng.randint(1, 200), rng.randint(1, 200)
            if a == b:
                continue
            expr = f"Which number is larger, {a} or {b}?"
            ans = f"{max(a, b)} is larger than {min(a, b)}."
        elif op == "alg":
            x = rng.randint(1, 30)
            c = rng.randint(1, 30)
            expr = f"Solve for x: x + {c} = {x + c}"
            ans = f"x = {x}"
        else:
            km = rng.randint(1, 50)
            expr = f"Convert {km} kilometres into metres."
            ans = f"{km} kilometres equals {km * 1000} metres."

        if expr.lower() in reserved_math or expr.lower() in HELDOUT_SET:
            continue
        rows.append({"turns": [(expr, ans)], "category": "math"})
        math_count += 1

    # 5. Out-of-scope refusal
    oos_qs = [
        "What is the stock price of Apple today?",
        "Predict tomorrow's lottery numbers.",
        "What is the weather in Tokyo right now?",
        "Give me a medical diagnosis for chest pain.",
        "What is the current price of Bitcoin?",
    ]
    for q in oos_qs:
        rows.append({"turns": [(q, OOS_RESPONSE)], "category": "out_of_scope"})

    return rows


def main():
    rng = random.Random(2026)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Fetching Simple Wikipedia (1,400 articles)...")
    wiki_rows = fetch_hf_rows("wikimedia/wikipedia", "20231101.simple", "train", total=1400, page_size=100)
    print("Fetching Dolly-15k (1,200 instructions)...")
    dolly_rows = fetch_hf_rows("databricks/databricks-dolly-15k", "default", "train", total=1200, page_size=100)
    print("Fetching ARC-Easy (600 science questions)...")
    arc_rows = fetch_hf_rows("allenai/ai2_arc", "ARC-Easy", "train", total=600, page_size=100)
    print("Fetching GSM8K (300 word problems)...")
    gsm_rows = fetch_hf_rows("openai/gsm8k", "main", "train", total=300, page_size=100)

    seen_hashes: Set[str] = set()
    pt_docs: List[dict] = []
    for w in wiki_rows:
        txt = (w.get("title", "") + "\n" + w.get("text", "")).strip()
        if len(txt) < 120:
            continue
        txt = txt[:1400].strip()
        h = hashlib.sha256(txt.encode("utf-8")).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            pt_docs.append({"text": txt, "source": "wikimedia/wikipedia:simple"})

    rng.shuffle(pt_docs)
    n_pt_val = max(60, int(0.08 * len(pt_docs)))
    pt_val = pt_docs[:n_pt_val]
    pt_train = pt_docs[n_pt_val:]

    sft_all: List[dict] = []
    seen_sft: Set[str] = set()

    for d in dolly_rows:
        inst = (d.get("instruction") or "").strip()
        ctx = (d.get("context") or "").strip()
        resp = (d.get("response") or "").strip()
        if not inst or not resp or len(resp) > 450:
            continue
        prompt = f"{inst}\nContext: {ctx[:400]}" if ctx else inst
        k = prompt.lower().strip()
        if k not in seen_sft and k not in HELDOUT_SET:
            seen_sft.add(k)
            sft_all.append({"turns": [(prompt, resp)], "category": f"dolly_{d.get('category', 'general')}"})

    for a in arc_rows:
        q = (a.get("question") or "").strip()
        ch = a.get("choices") or {}
        labels, texts = ch.get("label", []), ch.get("text", [])
        key = a.get("answerKey", "")
        if q and key in labels:
            ans = texts[labels.index(key)]
            k = q.lower().strip()
            if k not in seen_sft and k not in HELDOUT_SET:
                seen_sft.add(k)
                sft_all.append({"turns": [(q, ans)], "category": "arc_science"})

    for g in gsm_rows:
        q = (g.get("question") or "").strip()
        ans = (g.get("answer") or "").strip()
        k = q.lower().strip()
        if q and ans and len(ans) < 350 and k not in seen_sft and k not in HELDOUT_SET:
            seen_sft.add(k)
            sft_all.append({"turns": [(q, ans)], "category": "gsm8k"})

    for row in build_curated_general_sft(rng):
        k = row["turns"][0][0].lower().strip()
        if k not in seen_sft and k not in HELDOUT_SET:
            seen_sft.add(k)
            sft_all.append(row)

    rng.shuffle(sft_all)
    n_sft_val = max(100, int(0.08 * len(sft_all)))
    sft_val = sft_all[:n_sft_val]
    sft_train = sft_all[n_sft_val:]

    heldout = [{"input": p, "category": cat} for p, cat in HELDOUT_PROMPTS]

    files = {
        "pretrain_train.jsonl": pt_train,
        "pretrain_val.jsonl": pt_val,
        "sft_train.jsonl": sft_train,
        "sft_val.jsonl": sft_val,
        "heldout_eval.jsonl": heldout,
    }
    manifest = {"dataset": "clyffe_v012", "files": {}}
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

