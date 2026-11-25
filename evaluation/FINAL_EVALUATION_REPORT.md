# ARC-LB-01 Final Evaluation Report

**Checkpoint Evaluated:** `checkpoints/ARC-LB-01-FINAL_best.pt`
**SHA-256:** `3994a6dfff291f614b6d6353d0772e402707999b79d8f8784431cbfc1db7f832`

## 1. Capability & Boundary Results

| Category | Samples | Exact Accuracy | Semantic / Match Accuracy |
|---|---:|---:|---:|
| greetings | 100 | 24.00% | 48.00% |
| identity | 100 | 54.00% | 81.00% |
| conversation | 100 | 63.00% | 77.00% |
| science | 100 | 38.00% | 79.00% |
| general_knowledge | 100 | 64.00% | 78.00% |
| unseen_mathematics | 500 | 1.80% | 1.80% |
| out_of_scope | 200 | 69.00% | 69.00% |

## 2. Compression & Representation Audit

- **Dense Equivalent Params:** 246240
- **LittleBit Covered Params:** 245760 (99.81%)
- **Overall Effective BPW:** 0.6872
- **Checkpoint File Size:** 55713 bytes

## 3. Native Offline Inference Benchmark

- **Load Time:** 88.32 ms
- **Average Generation Latency:** 302.38 ms
- **Peak Python RAM:** 0.37 MB

## 4. Sample Outputs

- **User:** `Hello` -> **ARC-LB-01:** `Heree you later!`
- **User:** `Who are you?` -> **ARC-LB-01:** `I am ARC-LB-01, a native LittleBit-based experimental language model.`
- **User:** `What is gravity?` -> **ARC-LB-01:** `Gravity is the attractive force between objects that have mass.`
- **User:** `25 + 35` -> **ARC-LB-01:** `12`
- **User:** `What is the stock price of Apple today?` -> **ARC-LB-01:** `I don't have sufficient trained knowledge to answer this question.`
