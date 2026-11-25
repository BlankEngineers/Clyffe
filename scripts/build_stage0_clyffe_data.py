"""
Stage 0 Curated Dataset Builder for Clyffe v0.1 (research code ARC-LB-01).

Creates strictly disjoint train / validation / test / unseen_math / unseen_ood splits
in datasets/processed/stage_0/ with a SHA-256 + row-count manifest.json.
"""

import hashlib
import json
import os
import random
from typing import Dict, List, Set, Tuple


OOS_RESPONSE = "I don't have sufficient trained knowledge to answer this question."


def _expand_paraphrases(
    base_pairs: List[Tuple[str, str]],
    target_count: int,
    category: str,
    rng: random.Random,
) -> List[Dict[str, str]]:
    lead_ins = [
        "",
        "Please tell me, ",
        "Can you tell me: ",
        "Could you answer: ",
        "Quick question: ",
        "Tell me: ",
        "Hey, ",
        "Hi, ",
        "Excuse me, ",
        "I want to ask: ",
        "Please answer: ",
        "Do you know: ",
    ]
    endings = ["", "?", "."]
    rows: List[Dict[str, str]] = []
    seen: Set[str] = set()

    for inp, out in base_pairs:
        k = inp.strip().lower()
        if k not in seen:
            seen.add(k)
            rows.append({"input": inp.strip(), "output": out.strip(), "category": category})

    attempts = 0
    while len(rows) < target_count and attempts < target_count * 40:
        attempts += 1
        inp, out = rng.choice(base_pairs)
        lead = rng.choice(lead_ins)
        if lead and inp and inp[0].isalpha() and not inp.startswith(("ARC", "Clyffe", "I ")):
            cand = lead + inp[0].lower() + inp[1:]
        else:
            cand = lead + inp
        if not cand.endswith(("?", ".", "!")) and rng.random() < 0.5:
            cand += rng.choice(endings)
        k = cand.strip().lower()
        if k not in seen:
            seen.add(k)
            rows.append({"input": cand.strip(), "output": out.strip(), "category": category})

    rng.shuffle(rows)
    return rows


def build_stage0_splits(seed: int = 2026):
    rng = random.Random(seed)

    greetings = _expand_paraphrases(
        [
            ("Hello", "Hello! How can I help you?"),
            ("Hi", "Hi! How can I help you today?"),
            ("Hey", "Hey! How can I assist you?"),
            ("Good morning", "Good morning! How can I help you?"),
            ("Good afternoon", "Good afternoon! How can I assist you?"),
            ("Good evening", "Good evening! How can I help you?"),
            ("Good night", "Good night! Have a restful night."),
            ("How are you?", "I am doing well, thank you! How can I help you?"),
            ("Nice to meet you", "Nice to meet you too! I am Clyffe."),
            ("Thank you", "You are welcome!"),
            ("Goodbye", "Goodbye! Have a great day."),
        ],
        220,
        "greetings",
        rng,
    )

    identity = _expand_paraphrases(
        [
            ("Who are you?", "I am Clyffe, an experimental LittleBit-2 language model created by ARC III."),
            ("Who are u?", "I am Clyffe, an experimental LittleBit-2 language model created by ARC III."),
            ("What is your name?", "My name is Clyffe."),
            ("What model are you?", "I am Clyffe v0.1 (research code ARC-LB-01), a LittleBit-2 decoder-only model."),
            ("What is ARC-LB-01?", "ARC-LB-01 is the research project code name behind Clyffe."),
            ("Who built you?", "I was built by ARC III from random initialization using LittleBit-2."),
            ("Who developed you?", "I was developed by ARC III."),
            ("What is your model version?", "My model version is Clyffe v0.1."),
            ("What weight representation do you use?", "I use LittleBit-2 binary latent factorization with Joint-ITQ."),
            ("How were you trained?", "I was trained from random initialization using Quantization-Aware Training."),
        ],
        200,
        "identity",
        rng,
    )

    conversation = _expand_paraphrases(
        [
            ("What are you doing?", "I am processing your questions using LittleBit-2 latent computation."),
            ("Can you help me?", "Yes, I can help with greetings, identity, basic science, general knowledge, and simple math."),
            ("What can you do?", "I can answer greetings, identity questions, basic science, general facts, and simple arithmetic."),
            ("Are you ready?", "Yes, I am ready."),
            ("My name is Hari.", "Nice to meet you, Hari!"),
            ("My name is Priya.", "Nice to meet you, Priya!"),
            ("My name is Alex.", "Nice to meet you, Alex!"),
            ("Please help me.", "I will do my best to help with questions within my training scope."),
        ],
        200,
        "conversation",
        rng,
    )

    general_knowledge = _expand_paraphrases(
        [
            ("What is India?", "India is a country in South Asia and the world's most populous nation."),
            ("What is a computer?", "A computer is an electronic device that processes data and executes instructions."),
            ("What is artificial intelligence?", "Artificial intelligence is the field of building machines that perform tasks requiring human intelligence."),
            ("What is the internet?", "The internet is a global network connecting computers for communication and information sharing."),
            ("What is the Sun?", "The Sun is the star at the center of our solar system."),
            ("What is the Earth?", "Earth is the third planet from the Sun and supports life."),
            ("What is water?", "Water is a chemical compound made of hydrogen and oxygen, with formula H2O."),
            ("What is electricity?", "Electricity is the flow of electric charge, usually carried by electrons."),
        ],
        220,
        "general_knowledge",
        rng,
    )

    science = _expand_paraphrases(
        [
            ("What is gravity?", "Gravity is the attractive force between objects that have mass."),
            ("What is an atom?", "An atom is the basic unit of a chemical element."),
            ("What is an electron?", "An electron is a negatively charged subatomic particle that orbits an atomic nucleus."),
            ("What is energy?", "Energy is the capacity to do work or cause physical change."),
            ("What is light?", "Light is electromagnetic radiation visible to the human eye."),
            ("What is sound?", "Sound is a mechanical wave of pressure vibrations traveling through a medium."),
            ("What is photosynthesis?", "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and oxygen."),
            ("What is the solar system?", "The solar system consists of the Sun and all celestial bodies bound to it by gravity."),
        ],
        220,
        "science",
        rng,
    )

    # Arithmetic: train on 1..50, hold out 51..99 for unseen operand range generalization
    train_math: List[Dict[str, str]] = []
    unseen_math: List[Dict[str, str]] = []
    seen_math_inps: Set[str] = set()
    for _ in range(800):
        a, b = rng.randint(1, 50), rng.randint(1, 50)
        if rng.random() < 0.5:
            inp, out = f"{a} + {b}", str(a + b)
        else:
            hi, lo = max(a, b), min(a, b)
            inp, out = f"{hi} - {lo}", str(hi - lo)
        if inp not in seen_math_inps:
            seen_math_inps.add(inp)
            train_math.append({"input": inp, "output": out, "category": "mathematics"})

    for _ in range(250):
        a, b = rng.randint(51, 99), rng.randint(51, 99)
        if rng.random() < 0.5:
            inp, out = f"{a} + {b}", str(a + b)
        else:
            hi, lo = max(a, b), min(a, b)
            inp, out = f"{hi} - {lo}", str(hi - lo)
        if inp not in seen_math_inps:
            seen_math_inps.add(inp)
            unseen_math.append({"input": inp, "output": out, "category": "mathematics"})

    # In-distribution OOS prompts (seen topics) vs held-out OOD topics (never seen during training)
    oos_train_topics = [
        "the stock price of Apple today",
        "who won the FIFA World Cup in 2022",
        "a legal contract for a house lease",
        "how to diagnose chest pain",
        "the weather in Tokyo right now",
        "translating a paragraph into Japanese",
        "quantum chromodynamics renormalization group equations",
        "the mayor of Paris",
        "a recipe for sourdough bread",
        "tomorrow's lottery numbers",
    ]
    oos_ood_topics = [
        "the current price of Bitcoin",
        "who directed the movie Inception",
        "the capital of Brazil",
        "the GDP of Germany",
        "how to bake a chocolate cake",
        "the winner of last year's NBA finals",
        "the boiling point of mercury in Kelvin",
        "the population of Australia",
    ]
    tmpls = ["What is {}?", "Tell me about {}.", "Can you explain {}?", "Do you know {}?", "Explain {}."]

    oos_in = [
        {"input": t.format(top), "output": OOS_RESPONSE, "category": "out_of_scope"}
        for top in oos_train_topics
        for t in tmpls
    ]
    ood_heldout = [
        {"input": t.format(top), "output": OOS_RESPONSE, "category": "out_of_distribution"}
        for top in oos_ood_topics
        for t in tmpls
    ]

    all_train, all_val, all_test = [], [], []
    seen_global: Set[str] = set()

    for pool in (greetings, identity, conversation, general_knowledge, science, train_math, oos_in):
        deduped = []
        for r in pool:
            k = r["input"].strip().lower()
            if k not in seen_global:
                seen_global.add(k)
                deduped.append(r)
        rng.shuffle(deduped)
        n = len(deduped)
        nv = max(5, int(0.1 * n))
        nt = max(5, int(0.1 * n))
        all_val.extend(deduped[:nv])
        all_test.extend(deduped[nv : nv + nt])
        all_train.extend(deduped[nv + nt :])

    rng.shuffle(all_train)
    rng.shuffle(all_val)
    rng.shuffle(all_test)

    out_dir = "datasets/processed/stage_0"
    os.makedirs(out_dir, exist_ok=True)
    files = {
        "train.jsonl": all_train,
        "val.jsonl": all_val,
        "test.jsonl": all_test,
        "unseen_math.jsonl": unseen_math,
        "unseen_ood.jsonl": ood_heldout,
    }
    manifest = {"stage": "stage_0", "seed": seed, "files": {}}
    for fname, rows in files.items():
        fpath = os.path.join(out_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        h = hashlib.sha256(open(fpath, "rb").read()).hexdigest()
        manifest["files"][fname] = {"count": len(rows), "sha256": h}

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    build_stage0_splits()

