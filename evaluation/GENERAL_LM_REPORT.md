# Clyffe v0.2 — General Language Model Training Report (ARC-LB-01)

- **Checkpoint:** `checkpoints/clyffe-v0.2-general-20261125_best.pt` (`216347` bytes)
- **Architecture:** Decoder-only Transformer (6L, d=128, h=4, d_ff=384)
- **Dense-Equiv Params:** 2033280 | **LittleBit-2 Coverage:** 99.92% | **Effective BPW:** 0.5122
- **Tokenizer:** `vocab_size=4096`, `fertility=1.695`
- **Stage A LittleBit-2 Gradients All Non-Zero:** True (final loss `5.8923`)
- **Best SFT Val Loss:** `4.85306` | **Held-out 80-sample Avg F1:** `8.20%`
- **Avg Native Packed Latency:** `162.93 ms` | **Peak RAM:** `1.88 MB`

## Unseen Prompt Outputs

- **Explain gravity in simple English.** -> `The`
- **What is the difference between RAM and ROM?** -> `The first language`
- **Why does ice melt?** -> `The first tubs is a lot of the`
- **Convert 5 kilometres into metres.** -> `10 kilometres is 10000 metres.`
- **Calculate 345 + 678.** -> `110 + 101 = 110.`
- **What is the stock price of Apple today?** -> `The first common example of the first of the most as a way to be to the most.`

## Multi-Turn Conversation Check
- `My name is Hari.` -> `The first`
- `What is my name?` -> `The first goal is the first largest state in the`
- `Explain gravity.` -> `The`
