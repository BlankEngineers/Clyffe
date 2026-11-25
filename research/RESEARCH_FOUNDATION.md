# Research Foundation — ARC-LB-01

**Project:** ARC III — ARC-LB-01
**Research Title:** Native LittleBit-Based Tiny Language Model
**Date:** 2026-11-25

---

## 1. Research Paper References

1. **LittleBit (Original Method — NeurIPS 2025):**
   - **Title:** *LittleBit: Ultra Low-Bit Quantization via Latent Factorization*
   - **Authors:** Banseok Lee\*, Dongkyu Kim\*, Youngcheon You, Youngmin Kim (Samsung Research)
   - **Identifier:** [arXiv:2506.13771](https://arxiv.org/abs/2506.13771)
   - **Venue:** NeurIPS 2025

2. **LittleBit-2 (Follow-up Method — ICML 2026):**
   - **Title:** *LittleBit-2: Maximizing the Spectral Energy Gain in Sub-1-Bit LLMs via Latent Geometry Alignment*
   - **Authors:** Banseok Lee, Youngmin Kim (Samsung Research)
   - **Identifier:** [arXiv:2603.00042](https://arxiv.org/abs/2603.00042)
   - **Venue:** ICML 2026

3. **Official Reference Repository:**
   - **URL:** `https://github.com/SamsungLabs/LittleBit`

---

## 2. Licensing Restrictions

- The official `SamsungLabs/LittleBit` repository is licensed under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.
- **Implications for ARC-LB-01:**
  - All ARC-LB-01 code must be **independently written** from first principles based on the mathematical formulations in the papers and verified against the reference specification.
  - We must **not** copy the `SamsungLabs/LittleBit` repository wholesale.
  - Proper attribution to Banseok Lee, Dongkyu Kim, Youngcheon You, Youngmin Kim, and Samsung Research must be maintained in all research documentation and code headers where LittleBit / LittleBit-2 methods are referenced.
  - Because the underlying method's reference implementation carries CC BY-NC 4.0, ARC-LB-01 is strictly a non-commercial research initiative.

---

## 3. Mathematical Explanations & Exact Algorithms

### 3.1 Latent Factorization (LittleBit)

Given a dense weight matrix $W \in \mathbb{R}^{b \times a}$ (where $b = \text{out\_features}$, $a = \text{in\_features}$) and target latent rank $s = \text{split\_dim} \ll \min(a, b)$:

1. **Low-Rank SVD Initialization:**
   Compute rank-$s$ truncated SVD:
   $$W \approx U_t \Sigma_s V_t^T$$
   Distribute singular values symmetrically:
   $$U = U_t \Sigma_s^{1/2} \in \mathbb{R}^{b \times s}, \qquad V = \Sigma_s^{1/2} V_t^T \in \mathbb{R}^{s \times a}$$

2. **Dual-SVID (Rank-1 Scale Factorization on Absolute Values):**
   To compensate for binarization magnitude loss across rows, columns, and latent dimensions, apply rank-1 SVD to $|U|$ and $|V|$:
   $$|U| \approx u_1^T u_2, \quad u_1 \in \mathbb{R}^{1 \times b},\; u_2 \in \mathbb{R}^{1 \times s}$$
   $$|V| \approx v_1^T v_2, \quad v_1 \in \mathbb{R}^{1 \times s},\; v_2 \in \mathbb{R}^{1 \times a}$$
   where for a rank-1 SVD $|X| \approx \sigma_0 \tilde{u}_0 \tilde{v}_0^T$, the factors are $\sqrt{\sigma_0}\tilde{u}_0^T$ and $\sqrt{\sigma_0}\tilde{v}_0^T$.

3. **Binarization:**
   $$U_q = \text{sign}(U) \in \{-1, +1\}^{b \times s}, \qquad V_q = \text{sign}(V) \in \{-1, +1\}^{s \times a}$$
   (with $\text{sign}(0) \mapsto +1$).

4. **Weight Reconstruction:**
   $$\hat{W} = \big(U_q \odot (u_1^T u_2)\big)\big(V_q \odot (v_1^T v_2)\big)$$

5. **Optional Residual Branch (`residual=True`):**
   Compute residual matrix $W_R = W - \hat{W}$ and apply the exact same factorization + binarization + Dual-SVID to obtain $(U_{R,q}, V_{R,q}, u_{1,R}, u_{2,R}, v_{1,R}, v_{2,R})$ so that:
   $$\hat{W}_{\text{total}} = \hat{W} + \hat{W}_R$$

---

### 3.2 LittleBit-2: Internal Latent Rotation & Joint-ITQ

In LittleBit-1, SVD singular vectors $U, V$ often exhibit high coherence ("spiky" coordinate distributions) misaligned with the binary hypercube $\{-1, +1\}^s$, causing severe quantization error.

LittleBit-2 exploits rotational invariance of the latent product: for any orthogonal matrix $R \in O(s)$ ($R^T R = I_s$),
$$U V = (U R)(R^T V)$$

**Joint Iterative Quantization (Joint-ITQ):**
Form the stacked latent matrix:
$$X_{\text{combined}} = \begin{bmatrix} U \\ V^T \end{bmatrix} \in \mathbb{R}^{(b + a) \times s}$$
Solve:
$$\min_{R \in O(s),\; B \in \{-1,+1\}^{(b+a) \times s}} \| B - X_{\text{combined}} R \|_F^2$$
Initialize $R \in O(s)$ as a random orthogonal matrix and iterate $N_{\text{iter}}$ times (e.g., 20–50):
1. **Binary code update:** $B = \text{sign}(X_{\text{combined}} R)$
2. **Orthogonal Procrustes update:** Compute SVD of $M = B^T X_{\text{combined}} = U_p \Sigma_p V_p^T$, then update $R = V_p U_p^T$.

After convergence, apply the rotation:
$$U' = U R, \qquad V' = R^T V$$
Dual-SVID is then computed on $|U'|$ and $|V'|$. Because $R$ is absorbed into $U'$ and $V'$ at initialization, **LittleBit-2 adds zero inference-time overhead**.

---

### 3.3 Quantization-Aware Training (QAT)

During QAT, full-precision latent parameters $(U, V, u_1, u_2, v_1, v_2)$ are maintained and updated by the optimizer. On each forward pass, $U$ and $V$ are binarized using a surrogate-gradient function:
- **STEBinary:** Forward $\text{sign}(x)$ ($0 \mapsto +1$); backward $\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \mathbb{I}_{(-1, 1)}(x)$.
- **SmoothSign (default in LittleBit):** Forward $\text{sign}(x)$ ($0 \mapsto +1$); backward:
  $$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \alpha \big(1 - \tanh^2(\alpha x)\big), \quad \alpha = 100$$

---

### 3.4 Native Inference Computation (No Dense Reconstruction)

For an input activation $x \in \mathbb{R}^{N \times a}$, the factorized linear layer computes:
$$y = \Big(\big(((x \odot v_2) V_q^T) \odot (v_1 \odot u_2)\big) U_q^T\Big) \odot u_1$$
Notice that $v_1 \odot u_2 \in \mathbb{R}^{1 \times s}$ is fused into a single $s$-dimensional latent scale vector. If `residual=True`, the same sequence is computed for $(V_{R,q}, U_{R,q}, v_{2,R}, v_{1,R} \odot u_{2,R}, u_{1,R})$ and summed, followed by optional bias addition.

---

### 3.5 Memory Accounting

For a weight matrix of shape $(b \times a)$ with latent rank $s$:
- **Binary latent factors ($U_q, V_q$):** $s(a + b)$ bits, packed row-major into `int32` words (LSB-first, padded to 32-bit boundaries).
- **16-bit scale vectors ($u_1, v_2, v_1 \odot u_2$):** $16(a + b + s)$ bits.
- **Total bits (non-residual):**
  $$\text{Bits}_{\text{non-res}} = s(a + b + 16) + 16(a + b)$$
- **Total bits (residual):**
  $$\text{Bits}_{\text{res}} = 2s(a + b + 16) + 32(a + b)$$
- **Actual Effective Bits Per Weight (BPW):**
  $$\text{BPW}_{\text{actual}} = \frac{\text{Bits}}{a \cdot b}$$
  *(Note: For tiny matrices in a 100K–300K parameter model, $16(a+b)$ scale overhead is proportionally larger than in 7B-parameter models, so actual packed storage and BPW must be measured directly.)*

---

## 4. Original Authors' Contributions vs. ARC III Contributions

### Original Authors (Lee et al., Samsung Research)
- Formulated sub-1-bit latent factorization + Dual-SVID multi-scale compensation (LittleBit).
- Formulated Internal Latent Rotation + Joint-ITQ for latent geometry alignment (LittleBit-2).
- Demonstrated SmoothSign QAT and bit-packed inference kernels for **compressing pretrained FP16 LLMs** (OPT, Llama, Phi, Qwen, Gemma) using **Knowledge Distillation** (`KDTrainer` with a pretrained teacher model), while **excluding** `lm_head` and token embeddings from LittleBit factorization.

### ARC III Proposed Contributions (ARC-LB-01)
1. **From-Scratch Random Initialization Investigation:** Applying LittleBit-2 to randomly initialized weights (no pretrained model, no teacher model, no knowledge distillation).
2. **Pure Next-Token Prediction QAT:** Training a native LittleBit-2 language model solely via autoregressive cross-entropy loss.
3. **Full-Model Parameter Audit:** Investigating LittleBit-2 representation not only on attention/MLP linear projections, but also evaluating its feasibility on token embeddings and output projection (`lm_head`).
4. **Tiny-Scale Regime Analysis:** Measuring actual bit-packing overhead, spectral energy behavior on random matrices (Marchenko-Pastur vs. pretrained power-law spectra), and generalization on structured tasks (including unseen arithmetic) in the 100K–300K parameter regime.

---

## 5. Technical Limitations & Unresolved Research Questions

1. **Spectral Flatness of Random Initialization:** Pretrained weight matrices have fast-decaying singular values (high spectral energy in top-$s$ singular vectors). Randomly initialized matrices have flat Marchenko-Pastur singular value spectra, meaning a rank-$s$ SVD captures much less energy at initialization. How does Joint-ITQ + SmoothSign QAT behave when $U, V$ must *learn* structure from scratch rather than *preserve* pretrained structure?
2. **Small-Matrix Scale Overhead:** When $a, b \in [64, 256]$, the 16-bit scale vectors $16(a+b+s)$ represent a significant fraction of total bits compared to $a, b = 4096$. Thus, achieving $< 1.0$ actual BPW in tiny matrices requires careful rank selection and honest memory accounting.
3. **Gradient Flow Through SmoothSign:** With $\alpha = 100$, $\alpha(1 - \tanh^2(\alpha x))$ is a sharp spike around $x = 0$ and nearly zero for $|x| > 0.05$. If random weights or SVD factors have $|U_{ij}| > 0.05$, gradients vanish unless initialization scale, learning rate, or $\alpha$ scheduling is carefully tuned.
4. **Embeddings & Output Head:** In standard LittleBit, `lm_head` and `nn.Embedding` are excluded. Can `lm_head` (and/or tied embeddings) be represented via LittleBit-2 without collapsing next-token prediction accuracy on a tiny vocabulary?

