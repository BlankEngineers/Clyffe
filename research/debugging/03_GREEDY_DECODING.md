# Check 3 — Deterministic Greedy Decoding (Using `clyffe_v0.1_tokenizer`)

| Prompt | A | B | C-smoothsign | C-ste |
|---|---|---|---|---|
| `hello` | `'5'` (`[11]`) | `'5'` (`[11]`) | `'Hello! How can I help you?'` | `'You are welcome!'` |
| `hi` | `'Good morning! How can I help you?'` | `'3'` (`[10]`) | `'Hi! How can I help you today?'` | `'I was built by ARC III...'` |
| `who are you` | `'I am Clyffe, an experimental LittleBit-2 language model created by ARC III.'` | `'I am doing well,est to answer...'` | `'help you amYou are welcome!'` | `'I am doing well, thank you!...'` |
| `what is your name` | `'Myic!'` | `'Yes, I am mass.'` | `'Nice to meet you, Alex!'` | `'My name is Clyffe.'` |

**Findings:** With the correct tokenizer (`clyffe_v0.1_tokenizer`), outputs are **not** identical across checkpoints. The identical `"ello 6Hey f nqu th:"` gibberish seen in the UI occurred because `frontend/server.py` loaded the wrong tokenizer (`checkpoints/tokenizer`, `vocab=512`). However, `A` still fails on lowercase `"hello"` (`'5'`) and `"what is your name"` (`'Myic!'`) due to case-sensitive BPE tokenization and missing lowercase training coverage.

