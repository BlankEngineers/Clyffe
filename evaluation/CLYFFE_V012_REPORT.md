# Clyffe v0.1.2 — Diagnosis, Repair & General LM Report (ARC-LB-01)

- **Architecture:** 4L, `d_model=128`, `n_heads=4`, `d_ff=256`, `ctx=192`, `vocab=2048`
- **Exp B (20-example LittleBit-2 Overfit):** grad_all_nonzero=True, final_loss=4.1705, tf_acc=0.0797

## SmoothSign vs STEBinary Comparison

| Variant | Best SFT Val Loss | Final Train Loss | Dense Params | LB2 Coverage | Actual BPW | Disk Bytes | Avg Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1_SmoothSign | 4.22636 | 3.9691 | 1049728 | 99.89% | 0.557 | 133537 | 92.84 ms |
| C2_STEBinary | 4.18962 | 3.9888 | 1049728 | 99.89% | 0.557 | 132347 | 212.61 ms |

## Phase 11 Held-Out Prompt Outputs (C1_SmoothSign)

- **[conversation]** `Hello.` -> `Ne`
- **[conversation]** `How are you?` -> `There are the largest of the largest day of the same amount of the sameationale.`
- **[conversation]** `I am feeling happy.` -> `Clone is a total of the largest of the largest of the largest years of the largest years of the largest countries.`
- **[conversation]** `I am not in a good mood.` -> `Cefa is a bought to the most common a pinks.`
- **[conversation]** `Can you help me?` -> `Aley is a day of the same amount of the sameonomer of the same amount of the samething the same amount of the sameest of the sun.`
- **[conversation]** `Tell me something interesting.` -> `Aley is a total of the largest of the largest of the world.`
- **[general_knowledge]** `What is the solar system?` -> `The largest kilometres equals a day of the same amount of the sameonom of the sameonomical of the sameonomication of the sameonomical of the same amount of the sameonomication.`
- **[general_knowledge]** `Why does the moon shine?` -> `Clon is a day of the same amount of the same amount of the same amount of the bars.`
- **[general_knowledge]** `How does rain form?` -> `There are the same amount of the sameationale of the sameationale.`
- **[general_knowledge]** `What is the purpose of the human heart?` -> `Clon`
- **[science]** `Explain gravity.` -> `Clon is a day of the same amount of the same amount of the bars.`
- **[science]** `What is an atom?` -> `The United States of the Africi of the Africi of the Africi of the Africi of the Africi of the Africi of the Africi of the Orai, which is a stric of the Africi of`
- **[science]** `How does photosynthesis work?` -> `Cllon is a day of the same amount of the same amount of the same amount of the same level of the sameonomication.`
- **[science]** `Why does ice melt?` -> `There are a total of the same of the same amount of the same amount of the same amount of the bars.`
- **[science]** `What happens when water boils?` -> `Cle is a form of the same of the best the a total of the same of the same amount of the same of the same of the same of the same of the same of the same of the same of the same quest`
- **[technology]** `What is a computer?` -> `The United States of Germany is a year.`
- **[technology]** `Explain RAM and ROM.` -> `Aleon`
- **[technology]** `What is artificial intelligence?` -> `The United States of East East East East East East East East East East East East East East East East East East East East East East East East East East East East East East East East East East East East East East East East East East East East`
- **[technology]** `How does the internet work?` -> `Ixy is a total of the same amount of the sameonomical of the same amount of the sameonomically a total of the same amount of the same amount of the same amount of the bars.`
- **[mathematics]** `2*2` -> `1`
- **[mathematics]** `12*3` -> `11`
- **[mathematics]** `25+35` -> `188`
- **[mathematics]** `345+678` -> `128`
- **[mathematics]** `91-36` -> `198`
- **[mathematics]** `225/15` -> `188`
- **[mathematics]** `17*9` -> `11`
- **[instructions]** `Explain gravity in simple English.` -> `Clon is a day of the same amount of the history.`
- **[instructions]** `Write a paragraph about computers.` -> `In the Sun`
- **[instructions]** `Summarize a short passage: Plants convert sunlight, water, and carbon dioxide into food and oxygen through photosynthesis.` -> `The Barla, the Barla, Pryl, Ploy, and Pinger, and Pinger, and the Barrii, Crrocy,`
- **[instructions]** `Compare two planets: Earth and Mars.` -> `The Great Germany are very common of the Africi of the Africi of the Africi of the Africi of the Africi of the Africi of the Africi of the Africi of the Africi of the Africi of the`
- **[instructions]** `Explain the water cycle step by step.` -> `Clon is a bars of the bars of the most common a pinks.`
