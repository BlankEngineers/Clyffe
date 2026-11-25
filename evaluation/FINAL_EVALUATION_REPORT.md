# ARC-LB-01 Final Evaluation Report

**Checkpoint Evaluated:** `checkpoints/ARC-LB-01-V2_best.pt`
**SHA-256:** `d983f61c72696c45003e50b1d72e466e40e9f3d9dd25ee5f112ef98801f27cfe`

## 1. Capability & Boundary Results

| Category | Samples | Exact Accuracy | Semantic / Match Accuracy |
|---|---:|---:|---:|
| greetings | 48 | 93.75% | 97.92% |
| identity | 28 | 57.14% | 96.43% |
| conversation | 50 | 94.00% | 96.00% |
| science | 28 | 75.00% | 92.86% |
| general_knowledge | 28 | 67.86% | 78.57% |
| unseen_mathematics | 500 | 2.00% | 2.00% |
| out_of_scope | 200 | 97.00% | 97.00% |

## 2. Compression & Representation Audit

- **Dense Equivalent Params:** 246240
- **LittleBit Covered Params:** 245760 (99.81%)
- **Overall Effective BPW:** 0.8213
- **Checkpoint File Size:** 58247 bytes

## 3. Native Offline Inference Benchmark

- **Load Time:** 26.52 ms
- **Average Generation Latency:** 48.22 ms
- **Peak Python RAM:** 0.37 MB

## 4. Sample Outputs

- **User:** `Hello` -> **ARC-LB-01:** `Hello! How can I help you?`
- **User:** `Who are you?` -> **ARC-LB-01:** `I am Tesla 1.1 (ARC-LB-01), a native LittleBit-2 experimental language model.`
- **User:** `What is gravity?` -> **ARC-LB-01:** `Gravity is the attractive force between objects that have mass.`
- **User:** `25 + 35` -> **ARC-LB-01:** `18`
- **User:** `What is the stock price of Apple today?` -> **ARC-LB-01:** `I don't have sufficient trained knowledge to answer this question.`
