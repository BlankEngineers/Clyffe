# Mathematical Method Specification — ARC-LB-01

## 1. LittleBit-2 Latent Factorization
For a linear weight matrix $W \in \mathbb{R}^{b \times a}$ with latent rank $s$:
$$W \approx U V, \quad U = U_t \Sigma_s^{1/2} \in \mathbb{R}^{b \times s}, \quad V = \Sigma_s^{1/2} V_t^T \in \mathbb{R}^{s \times a}$$

## 2. Internal Latent Rotation & Joint-ITQ
Using rotational invariance $UV = (UR)(R^T V)$ for $R \in O(s)$, Joint-ITQ minimizes:
$$\min_{R \in O(s),\, B \in \{-1,+1\}^{(b+a)\times s}} \| B - X_{\text{combined}} R \|_F^2, \quad X_{\text{combined}} = \begin{bmatrix} U \\ V^T \end{bmatrix}$$
via alternating sign quantization $B = \text{sign}(X_{\text{combined}} R)$ and orthogonal Procrustes SVD ($M = B^T X_{\text{combined}} = U_p \Sigma_p V_p^T$, $R = V_p U_p^T$).

## 3. Dual-SVID & Binarization
Rank-1 SVD on $|U|$ and $|V|$ yields $(u_1, u_2)$ and $(v_1, v_2)$. Binary latent factors are $U_q = \text{sign}(U)$, $V_q = \text{sign}(V)$.

## 4. Native Inference Forward Pass
Without reconstructing $\hat{W}$:
$$y = \Big(\big(((x \odot v_2) V_q^T) \odot (v_1 \odot u_2)\big) U_q^T\Big) \odot u_1$$

## 5. Adaptation for Random Initialization (ARC III)
1. **SmoothSign Sharpness Tuning:** In pretrained LLMs, $\alpha = 100$ works near pre-converged factors. From random initialization, $\alpha = 20$ widens the active surrogate gradient window so latent factors can cross zero during early training.
2. **Full-Model Coverage:** Extended LittleBit-2 beyond attention/MLP linear layers to `LittleBitEmbedding` and `lm_head` so >99% of learned parameters use binary latent representation.

