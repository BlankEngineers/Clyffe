"""
Original Dataset Generator for ARC-LB-01.

Generates ~5,500 training examples plus separate validation and test splits
(including strictly disjoint unseen arithmetic expressions) across:
- Greetings (500)
- Identity (500)
- Conversation (1,000)
- General knowledge (1,000)
- Science (1,000)
- Mathematics (1,000)
- Out-of-scope / Unknown (500)
"""

import json
import os
import random
from typing import Dict, List


OOS_RESPONSE = "I don't have sufficient trained knowledge to answer this question."


def _expand_templates(
    pairs: List[tuple],
    target_count: int,
    category: str,
    rng: random.Random,
) -> List[Dict[str, str]]:
    """Expand curated Q&A pairs with phrasing variations to reach target_count."""
    prefixes = ["", "Please ", "Tell me, ", "Can you tell me: ", "Hey, "]
    suffixes = ["", "?", "."]
    records: List[Dict[str, str]] = []
    seen = set()

    # First include all base pairs
    for inp, out in pairs:
        records.append({"input": inp, "output": out, "category": category})
        seen.add((inp.lower(), out))

    while len(records) < target_count:
        inp, out = rng.choice(pairs)
        pref = rng.choice(prefixes)
        if pref and inp and inp[0].isalpha():
            varied = pref + inp[0].lower() + inp[1:]
        else:
            varied = inp
        if not varied.endswith(("?", ".", "!")) and rng.random() < 0.4:
            varied = varied + rng.choice(suffixes)
        records.append({"input": varied, "output": out, "category": category})

    rng.shuffle(records)
    return records[:target_count]


def build_greetings(rng: random.Random, n: int = 500) -> List[Dict[str, str]]:
    pairs = [
        ("Hello", "Hello! How can I help you?"),
        ("Hi", "Hi! How can I help you today?"),
        ("Hey", "Hey! What can I help you with?"),
        ("Good morning", "Good morning! How can I assist you?"),
        ("Good afternoon", "Good afternoon! How can I help you?"),
        ("Good evening", "Good evening! How can I help you?"),
        ("Good night", "Good night! Have a restful night."),
        ("How are you?", "I am doing well, thank you! How can I help you?"),
        ("Nice to meet you", "Nice to meet you too! I am ARC-LB-01."),
        ("Hello there", "Hello! How can I assist you today?"),
        ("Hi there", "Hi! I am ready to help."),
        ("Greetings", "Greetings! How can I help you?"),
    ]
    return _expand_templates(pairs, n, "greetings", rng)


def build_identity(rng: random.Random, n: int = 500) -> List[Dict[str, str]]:
    pairs = [
        ("Who are you?", "I am ARC-LB-01, a native LittleBit-based experimental language model."),
        ("What is your name?", "My name is ARC-LB-01."),
        ("What model are you?", "I am ARC-LB-01, a tiny decoder-only LittleBit-2 language model."),
        ("What is ARC-LB-01?", "ARC-LB-01 is an original native LittleBit-2 language model built from scratch."),
        ("What does ARC III mean?", "ARC III is the research initiative developing ARC-LB-01."),
        ("Who built you?", "I was developed by the ARC III research project."),
        ("Who developed you?", "I was developed under the ARC III research initiative."),
        ("Are you GPT or Llama?", "No. I am ARC-LB-01, independently initialized and trained with LittleBit-2."),
        ("What weight representation do you use?", "I use LittleBit-2 binary latent factorization with Joint-ITQ."),
        ("How were you trained?", "I was trained from random initialization using Quantization-Aware Training."),
    ]
    return _expand_templates(pairs, n, "identity", rng)


def build_conversation(rng: random.Random, n: int = 1000) -> List[Dict[str, str]]:
    pairs = [
        ("How are you?", "I am functioning well and ready to assist you."),
        ("What are you doing?", "I am processing your questions using LittleBit-2 latent computation."),
        ("Can you help me?", "Yes, I can help with greetings, identity, basic science, general knowledge, and simple math."),
        ("Thank you", "You are welcome!"),
        ("Thanks", "You are welcome! Let me know if you need anything else."),
        ("Goodbye", "Goodbye! Have a great day."),
        ("See you later", "See you later!"),
        ("Bye", "Goodbye!"),
        ("What can you do?", "I can answer basic greetings, identity questions, simple science, general facts, and basic arithmetic."),
        ("Are you ready?", "Yes, I am ready."),
        ("Okay", "Okay! What would you like to ask next?"),
        ("Great", "Glad to hear that!"),
        ("My name is Hari.", "Nice to meet you, Hari!"),
        ("My name is Priya.", "Nice to meet you, Priya!"),
        ("My name is Alex.", "Nice to meet you, Alex!"),
        ("Please help me.", "I will do my best to help with questions within my training scope."),
    ]
    return _expand_templates(pairs, n, "conversation", rng)


def build_general_knowledge(rng: random.Random, n: int = 1000) -> List[Dict[str, str]]:
    pairs = [
        ("What is India?", "India is a country in South Asia and the world's most populous nation."),
        ("What is a computer?", "A computer is an electronic device that processes data and executes instructions."),
        ("What is artificial intelligence?", "Artificial intelligence is the field of building machines capable of performing tasks that typically require human intelligence."),
        ("What is the internet?", "The internet is a global network connecting millions of computers for communication and information sharing."),
        ("What is the Sun?", "The Sun is the star at the center of our solar system that provides light and heat to Earth."),
        ("What is the Earth?", "Earth is the third planet from the Sun and the only known planet supporting life."),
        ("What is water?", "Water is a clear chemical compound made of hydrogen and oxygen, with formula H2O."),
        ("What is electricity?", "Electricity is the flow of electric charge, usually carried by electrons."),
        ("What is a CPU?", "A CPU, or central processing unit, is the primary processor that executes computer instructions."),
        ("What is memory in a computer?", "Computer memory stores data and instructions for fast access by the processor."),
        ("What is binary?", "Binary is a base-2 number system that represents data using only 0 and 1."),
        ("What is a language model?", "A language model is a statistical or neural system trained to predict and generate sequences of tokens."),
    ]
    return _expand_templates(pairs, n, "general_knowledge", rng)


def build_science(rng: random.Random, n: int = 1000) -> List[Dict[str, str]]:
    pairs = [
        ("What is gravity?", "Gravity is the attractive force between objects that have mass."),
        ("What is an atom?", "An atom is the basic unit of a chemical element."),
        ("What is an electron?", "An electron is a negatively charged subatomic particle that orbits an atomic nucleus."),
        ("What is energy?", "Energy is the capacity to do work or cause physical change."),
        ("What is light?", "Light is electromagnetic radiation visible to the human eye."),
        ("What is sound?", "Sound is a mechanical wave of pressure vibrations traveling through a medium."),
        ("What is photosynthesis?", "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to produce glucose and oxygen."),
        ("What is the solar system?", "The solar system consists of the Sun and all celestial bodies bound to it by gravity, including eight planets."),
        ("What is mass?", "Mass is a measure of the amount of matter in an object."),
        ("What is a proton?", "A proton is a positively charged subatomic particle found in the nucleus of an atom."),
        ("What is a neutron?", "A neutron is an electrically neutral subatomic particle in the atomic nucleus."),
        ("What is temperature?", "Temperature is a measure of the average kinetic energy of particles in a substance."),
    ]
    return _expand_templates(pairs, n, "science", rng)


def _format_math_problem(op: str, rng: random.Random) -> tuple:
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
    # average of 3 numbers divisible by 3
    m = rng.randint(2, 30)
    d1 = rng.randint(-m + 1, 10)
    d2 = rng.randint(-m + 1, 10)
    x1, x2, x3 = m + d1, m + d2, m - d1 - d2
    if min(x1, x2, x3) < 0:
        x1, x2, x3 = 10, 20, 30
        m = 20
    return f"Average of {x1}, {x2} and {x3}", str(m)


def build_math_splits(rng: random.Random, train_n: int = 1000, unseen_n: int = 500):
    ops = ["+", "-", "*", "/", "avg"]
    train_records: List[Dict[str, str]] = []
    seen_inputs = set()

    # Include canonical prompt examples in train
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
        if inp not in seen_inputs:
            seen_inputs.add(inp)
            train_records.append({"input": inp, "output": out, "category": "mathematics"})

    unseen_records: List[Dict[str, str]] = []
    while len(unseen_records) < unseen_n:
        op = rng.choice(ops)
        inp, out = _format_math_problem(op, rng)
        if inp not in seen_inputs:
            seen_inputs.add(inp)
            unseen_records.append({"input": inp, "output": out, "category": "mathematics"})

    return train_records, unseen_records


def build_out_of_scope(rng: random.Random, n: int = 500) -> List[Dict[str, str]]:
    prompts = [
        "What is the stock price of Apple today?",
        "Who won the FIFA World Cup in 2022?",
        "Write a legal contract for a house lease.",
        "Diagnose my chest pain.",
        "What is the weather in Tokyo right now?",
        "Translate this paragraph into Japanese.",
        "Explain quantum chromodynamics renormalization group equations.",
        "Who is the mayor of Paris?",
        "Give me a recipe for sourdough bread.",
        "Predict tomorrow's lottery numbers.",
        "Summarize the novel War and Peace.",
        "What is the latest news in politics?",
        "How do I hack into a Wi-Fi network?",
        "Write a 500-word essay on Roman architecture.",
        "What is the capital of Brazil?",
        "What is the GDP of Germany?",
    ]
    records: List[Dict[str, str]] = []
    while len(records) < n:
        p = rng.choice(prompts)
        records.append({"input": p, "output": OOS_RESPONSE, "category": "out_of_scope"})
    return records


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
        build_greetings(rng, 500),
        build_identity(rng, 500),
        build_conversation(rng, 1000),
        build_general_knowledge(rng, 1000),
        build_science(rng, 1000),
        build_out_of_scope(rng, 500),
    ]
    math_train, math_unseen = build_math_splits(rng, train_n=1000, unseen_n=500)
    categories.append(math_train)

    for cat_rows in categories:
        rng.shuffle(cat_rows)
        n = len(cat_rows)
        n_val = max(1, int(0.1 * n))
        n_test = max(1, int(0.1 * n))
        all_val.extend(cat_rows[:n_val])
        all_test.extend(cat_rows[n_val : n_val + n_test])
        all_train.extend(cat_rows[n_val + n_test :])

    rng.shuffle(all_train)
    rng.shuffle(all_val)
    rng.shuffle(all_test)

    save_jsonl("datasets/train/train.jsonl", all_train)
    save_jsonl("datasets/validation/val.jsonl", all_val)
    save_jsonl("datasets/test/test.jsonl", all_test)
    save_jsonl("datasets/test/unseen_math.jsonl", math_unseen)

    summary = {
        "train_count": len(all_train),
        "val_count": len(all_val),
        "test_count": len(all_test),
        "unseen_math_count": len(math_unseen),
        "total_main_pool": len(all_train) + len(all_val) + len(all_test),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

