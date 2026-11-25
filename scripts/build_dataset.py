"""
Improved Disjoint Dataset Generator for ARC-LB-01 / Tesla 1.1 (v0.2.0).

Fixes from v0.1.0 audit:
1. Strictly disjoint train / validation / test splits (zero exact input overlap).
2. Complete coverage of all required greetings, identity, conversation,
   science, general knowledge, mathematics, and out-of-scope categories.
3. Diverse out-of-scope phrasings to prevent phrase leakage.
"""

import json
import os
import random
from typing import Dict, List, Set, Tuple


OOS_RESPONSE = "I don't have sufficient trained knowledge to answer this question."


def _build_unique_variations(
    base_pairs: List[Tuple[str, str]],
    target_count: int,
    category: str,
    rng: random.Random,
) -> List[Dict[str, str]]:
    """Generate unique (input, output) pairs without duplicating identical inputs."""
    lead_ins = [
        "",
        "Please tell me, ",
        "Can you answer: ",
        "Could you explain: ",
        "I want to ask: ",
        "Tell me: ",
        "Hey, ",
        "Hi, ",
        "Excuse me, ",
        "Quick question: ",
        "Please answer ",
        "Do you know: ",
    ]
    endings = ["", "?", "."]

    records: List[Dict[str, str]] = []
    seen_inputs: Set[str] = set()

    for inp, out in base_pairs:
        key = inp.strip().lower()
        if key not in seen_inputs:
            seen_inputs.add(key)
            records.append({"input": inp.strip(), "output": out.strip(), "category": category})

    attempts = 0
    max_attempts = target_count * 40
    while len(records) < target_count and attempts < max_attempts:
        attempts += 1
        inp, out = rng.choice(base_pairs)
        lead = rng.choice(lead_ins)
        if lead and inp and inp[0].isalpha() and not inp.startswith(("ARC", "I ")):
            candidate = lead + inp[0].lower() + inp[1:]
        else:
            candidate = lead + inp
        if not candidate.endswith(("?", ".", "!")) and rng.random() < 0.5:
            candidate += rng.choice(endings)
        key = candidate.strip().lower()
        if key not in seen_inputs:
            seen_inputs.add(key)
            records.append({"input": candidate.strip(), "output": out.strip(), "category": category})

    rng.shuffle(records)
    return records


def build_greetings(rng: random.Random, n: int = 500) -> List[Dict[str, str]]:
    pairs = [
        ("Hello", "Hello! How can I help you?"),
        ("Hi", "Hi! How can I help you today?"),
        ("Hey", "Hey! How can I assist you?"),
        ("Good morning", "Good morning! How can I help you?"),
        ("Good afternoon", "Good afternoon! How can I assist you?"),
        ("Good evening", "Good evening! How can I help you?"),
        ("Good night", "Good night! Have a restful night."),
        ("How are you?", "I am doing well, thank you! How can I help you?"),
        ("Nice to meet you", "Nice to meet you too! I am Tesla."),
        ("Thank you", "You are welcome!"),
        ("Goodbye", "Goodbye! Have a great day."),
        ("Hello there", "Hello! How can I assist you today?"),
        ("Hi there", "Hi! I am ready to help."),
        ("Greetings", "Greetings! How can I help you?"),
        ("Thanks", "You are welcome!"),
        ("See you later", "See you later!"),
    ]
    return _build_unique_variations(pairs, n, "greetings", rng)


def build_identity(rng: random.Random, n: int = 500) -> List[Dict[str, str]]:
    pairs = [
        ("Who are you?", "I am Tesla made by Blank Engineers."),
        ("Who are u?", "I am Tesla made by Blank Engineers."),
        ("who are u", "I am Tesla made by Blank Engineers."),
        ("What is your name?", "My name is Tesla."),
        ("What model are you?", "I am Tesla, a tiny decoder-only LittleBit-2 language model made by Blank Engineers."),
        ("What is ARC-LB-01?", "ARC-LB-01 is the experimental identifier for Tesla, built with LittleBit-2."),
        ("What is ARC III?", "ARC III is the research ecosystem under which Blank Engineers developed Tesla."),
        ("What does ARC III mean?", "ARC III is the research initiative developing native LittleBit-2 language models."),
        ("Who built you?", "I am Tesla made by Blank Engineers under the ARC III research ecosystem."),
        ("Who developed you?", "I was developed by Vishnupriyan Sivanesan and Chittari Hari Dhanush at Blank Engineers."),
        ("What is your model version?", "My model version is 0.3.0 (Tesla / ARC-LB-01)."),
        ("Are you GPT or Llama?", "No. I am Tesla made by Blank Engineers, trained from random initialization with LittleBit-2."),
        ("What weight representation do you use?", "I use LittleBit-2 binary latent factorization with Joint-ITQ."),
        ("How were you trained?", "I was trained from random initialization using Quantization-Aware Training."),
    ]
    return _build_unique_variations(pairs, n, "identity", rng)


def build_conversation(rng: random.Random, n: int = 1000) -> List[Dict[str, str]]:
    pairs = [
        ("How are you?", "I am functioning well and ready to assist you."),
        ("What are you doing?", "I am processing your questions using LittleBit-2 latent computation."),
        ("Can you help me?", "Yes, I can help with greetings, identity, basic science, general facts, and simple math."),
        ("Thank you", "You are welcome!"),
        ("Thanks a lot", "You are welcome! Let me know if you need anything else."),
        ("Goodbye", "Goodbye! Have a great day."),
        ("See you later", "See you later!"),
        ("Bye", "Goodbye!"),
        ("What can you do?", "I can answer greetings, identity questions, basic science, general facts, and simple math."),
        ("Are you ready?", "Yes, I am ready."),
        ("Okay", "Okay! What would you like to ask next?"),
        ("Great", "Glad to hear that!"),
        ("My name is Hari.", "Nice to meet you, Hari!"),
        ("My name is Priya.", "Nice to meet you, Priya!"),
        ("My name is Alex.", "Nice to meet you, Alex!"),
        ("Please help me.", "I will do my best to help with questions within my training scope."),
    ]
    return _build_unique_variations(pairs, n, "conversation", rng)


def build_general_knowledge(rng: random.Random, n: int = 1000) -> List[Dict[str, str]]:
    pairs = [
        ("What is India?", "India is a country in South Asia and the world's most populous nation."),
        ("What is a computer?", "A computer is an electronic device that processes data and executes instructions."),
        ("What is artificial intelligence?", "Artificial intelligence is the field of building machines that perform tasks requiring human intelligence."),
        ("What is the internet?", "The internet is a global network connecting computers for communication and information sharing."),
        ("What is the Sun?", "The Sun is the star at the center of our solar system that provides light and heat to Earth."),
        ("What is the Earth?", "Earth is the third planet from the Sun and the only known planet supporting life."),
        ("What is water?", "Water is a clear chemical compound made of hydrogen and oxygen, with formula H2O."),
        ("What is electricity?", "Electricity is the flow of electric charge, usually carried by electrons."),
        ("What is a CPU?", "A CPU, or central processing unit, is the primary processor that executes computer instructions."),
        ("What is memory in a computer?", "Computer memory stores data and instructions for fast access by the processor."),
        ("What is binary?", "Binary is a base-2 number system that represents data using only 0 and 1."),
        ("What is a language model?", "A language model is a neural or statistical system trained to predict sequences of tokens."),
    ]
    return _build_unique_variations(pairs, n, "general_knowledge", rng)


def build_science(rng: random.Random, n: int = 1000) -> List[Dict[str, str]]:
    pairs = [
        ("What is gravity?", "Gravity is the attractive force between objects that have mass."),
        ("What is an atom?", "An atom is the basic unit of a chemical element."),
        ("What is an electron?", "An electron is a negatively charged subatomic particle that orbits an atomic nucleus."),
        ("What is energy?", "Energy is the capacity to do work or cause physical change."),
        ("What is light?", "Light is electromagnetic radiation visible to the human eye."),
        ("What is sound?", "Sound is a mechanical wave of pressure vibrations traveling through a medium."),
        ("What is photosynthesis?", "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and oxygen."),
        ("What is the solar system?", "The solar system consists of the Sun and all celestial bodies bound to it by gravity."),
        ("What is mass?", "Mass is a measure of the amount of matter in an object."),
        ("What is a proton?", "A proton is a positively charged subatomic particle found in the nucleus of an atom."),
        ("What is a neutron?", "A neutron is an electrically neutral subatomic particle in the atomic nucleus."),
        ("What is temperature?", "Temperature is a measure of the average kinetic energy of particles in a substance."),
    ]
    return _build_unique_variations(pairs, n, "science", rng)


def _format_math_problem(op: str, rng: random.Random) -> Tuple[str, str]:
    if op == "+":
        a, b = rng.randint(1, 99), rng.randint(1, 99)
        return f"{a} + {b}", str(a + b)
    if op == "-":
        a = rng.randint(10, 150)
        b = rng.randint(1, a)
        return f"{a} - {b}", str(a - b)
    if op == "*":
        a, b = rng.randint(2, 20), rng.randint(2, 20)
        sym = rng.choice(["*", "×"])
        return f"{a} {sym} {b}", str(a * b)
    if op == "/":
        b = rng.randint(2, 15)
        ans = rng.randint(2, 20)
        a = b * ans
        sym = rng.choice(["/", "÷"])
        return f"{a} {sym} {b}", str(ans)
    m = rng.randint(2, 30)
    d1 = rng.randint(-m + 1, 10)
    d2 = rng.randint(-m + 1, 10)
    x1, x2, x3 = m + d1, m + d2, m - d1 - d2
    if min(x1, x2, x3) < 0:
        x1, x2, x3 = 10, 20, 30
        m = 20
    return f"Average of {x1}, {x2} and {x3}", str(m)


def build_math_splits(rng: random.Random, train_n: int = 1200, unseen_n: int = 500):
    ops = ["+", "-", "*", "/", "avg"]
    train_records: List[Dict[str, str]] = []
    seen_inputs = set()

    # Reserve audit test expressions so they stay strictly unseen
    reserved = {"37 + 48", "91 - 36", "17 × 9", "225 ÷ 15"}

    canonical = [
        ("25 + 35", "60"),
        ("100 - 45", "55"),
        ("12 × 8", "96"),
        ("144 ÷ 12", "12"),
        ("Average of 10, 20 and 30", "20"),
    ]
    for inp, out in canonical:
        train_records.append({"input": inp, "output": out, "category": "mathematics"})
        seen_inputs.add(inp)

    while len(train_records) < train_n:
        op = rng.choice(ops)
        inp, out = _format_math_problem(op, rng)
        if inp not in seen_inputs and inp not in reserved:
            seen_inputs.add(inp)
            train_records.append({"input": inp, "output": out, "category": "mathematics"})

    unseen_records: List[Dict[str, str]] = [
        {"input": "37 + 48", "output": "85", "category": "mathematics"},
        {"input": "91 - 36", "output": "55", "category": "mathematics"},
        {"input": "17 × 9", "output": "153", "category": "mathematics"},
        {"input": "225 ÷ 15", "output": "15", "category": "mathematics"},
    ]
    while len(unseen_records) < unseen_n:
        op = rng.choice(ops)
        inp, out = _format_math_problem(op, rng)
        if inp not in seen_inputs and inp not in reserved:
            seen_inputs.add(inp)
            unseen_records.append({"input": inp, "output": out, "category": "mathematics"})

    return train_records, unseen_records


def build_out_of_scope(rng: random.Random, n: int = 500) -> List[Dict[str, str]]:
    topics = [
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
        "a summary of War and Peace",
        "the latest news in politics",
        "the capital of Brazil",
        "the GDP of Germany",
        "the current price of Bitcoin",
        "who directed the movie Inception",
        "the population of Australia",
        "how to bake a chocolate cake",
        "the winner of last year's NBA finals",
        "the boiling point of mercury in Kelvin",
    ]
    templates = [
        "What is {}?",
        "Tell me about {}.",
        "Can you explain {}?",
        "Please provide {}.",
        "Do you know {}?",
        "Give me {}.",
        "I need {}.",
        "Explain {}.",
    ]
    records: List[Dict[str, str]] = []
    seen = set()
    for t in topics:
        for tmpl in templates:
            q = tmpl.format(t)
            if q.lower() not in seen:
                seen.add(q.lower())
                records.append({"input": q, "output": OOS_RESPONSE, "category": "out_of_scope"})
    rng.shuffle(records)
    return records[:n]


def save_jsonl(path: str, rows: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    rng = random.Random(2026)

    all_train: List[Dict[str, str]] = []
    all_val: List[Dict[str, str]] = []
    all_test: List[Dict[str, str]] = []

    categories = [
        build_greetings(rng, 240),
        build_identity(rng, 220),
        build_conversation(rng, 300),
        build_general_knowledge(rng, 260),
        build_science(rng, 260),
        build_out_of_scope(rng, 160),
    ]
    math_train, math_unseen = build_math_splits(rng, train_n=1200, unseen_n=500)
    categories.append(math_train)

    seen_global = set()
    for cat_rows in categories:
        deduped = []
        for r in cat_rows:
            k = r["input"].strip().lower()
            if k not in seen_global:
                seen_global.add(k)
                deduped.append(r)
        rng.shuffle(deduped)
        n = len(deduped)
        n_val = max(10, int(0.1 * n))
        n_test = max(10, int(0.1 * n))
        all_val.extend(deduped[:n_val])
        all_test.extend(deduped[n_val : n_val + n_test])
        all_train.extend(deduped[n_val + n_test :])

    rng.shuffle(all_train)
    rng.shuffle(all_val)
    rng.shuffle(all_test)

    # Verify zero overlap across splits
    train_keys = {(r["input"], r["output"]) for r in all_train}
    val_keys = {(r["input"], r["output"]) for r in all_val}
    test_keys = {(r["input"], r["output"]) for r in all_test}
    assert len(train_keys & val_keys) == 0, "Train/Val overlap detected!"
    assert len(train_keys & test_keys) == 0, "Train/Test overlap detected!"

    save_jsonl("datasets/train/train.jsonl", all_train)
    save_jsonl("datasets/validation/val.jsonl", all_val)
    save_jsonl("datasets/test/test.jsonl", all_test)
    save_jsonl("datasets/test/unseen_math.jsonl", math_unseen)

    summary = {
        "train_count": len(all_train),
        "val_count": len(all_val),
        "test_count": len(all_test),
        "unseen_math_count": len(math_unseen),
        "overlap_train_val": len(train_keys & val_keys),
        "overlap_train_test": len(train_keys & test_keys),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
