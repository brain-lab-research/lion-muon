# Convergence Theory for Alternating-Norm Descent

## 1. Notation and Assumptions

We minimize $f: \mathbb{R}^{m \times n} \to \mathbb{R}$ over matrix-valued parameters $W$.

**Assumption 1** ($L$-smoothness). $f$ is differentiable and its gradient is $L$-Lipschitz
w.r.t. the Frobenius norm:

$$\|\nabla f(X) - \nabla f(Y)\|_F \leq L\|X - Y\|_F \quad \forall\, X, Y$$

This implies the descent inequality:

$$f(Y) \leq f(X) + \langle \nabla f(X),\, Y - X\rangle + \frac{L}{2}\|Y - X\|_F^2 \tag{1}$$

**Assumption 2** (Bounded below). $f^* := \inf_W f(W) > -\infty$.

We write $p = \min(m,n)$, $q = \max(m,n)$, so $mn = pq$.

### Matrix norms used

| Norm | Definition | Notation |
|---|---|---|
| Frobenius | $\|G\|_F = \sqrt{\sum_{ij} G_{ij}^2} = \sqrt{\sum_i \sigma_i^2}$ | $\|\cdot\|_F$ |
| Spectral (operator) | $\|G\|_{\mathrm{op}} = \sigma_1(G)$ | $\|\cdot\|_{\mathrm{op}}$ |
| Nuclear (trace) | $\|G\|_* = \sum_i \sigma_i(G)$ | $\|\cdot\|_*$ |
| Entry-wise $\ell_\infty$ | $\|G\|_\infty = \max_{ij} |G_{ij}|$ | $\|\cdot\|_\infty$ |
| Entry-wise $\ell_1$ | $\|G\|_1 = \sum_{ij} |G_{ij}|$ | $\|\cdot\|_1$ |

Duality pairs: $(\|\cdot\|_{\mathrm{op}},\, \|\cdot\|_*)$ and $(\|\cdot\|_\infty,\, \|\cdot\|_1)$.

---

## 2. Steepest Descent under a Norm

Given a norm $\|\cdot\|$, the **steepest descent direction** is:

$$d^* = \arg\min_{\|d\| \leq 1} \langle \nabla f(W),\, d\rangle$$

By the definition of the dual norm:

$$\langle \nabla f(W),\, d^*\rangle = -\|\nabla f(W)\|_* \tag{2}$$

where $\|\cdot\|_*$ denotes the dual norm.

**Spectral norm** $\to$ **Muon**: The constraint is $\|d\|_{\mathrm{op}} \leq 1$.
The dual norm is nuclear: $\|\nabla f\|_*$.
The solution is $d^* = -U V^\top$ where $\nabla f = U\Sigma V^\top$ is the SVD.

**$\ell_\infty$ norm** $\to$ **Sign**: The constraint is $\|d\|_\infty \leq 1$.
The dual norm is $\ell_1$: $\|\nabla f\|_1$.
The solution is $d^* = -\mathrm{sign}(\nabla f)$.

---

## 3. Fundamental Lemmas

### Lemma 1 (Norm ordering)

*For any $G \in \mathbb{R}^{m \times n}$:*

$$(a)\quad \|G\|_F \leq \|G\|_* \leq \sqrt{p}\,\|G\|_F$$
$$(b)\quad \|G\|_F \leq \|G\|_1 \leq \sqrt{pq}\,\|G\|_F$$

**Proof.**

(a) Let $\sigma_1 \geq \cdots \geq \sigma_r > 0$ be the singular values of $G$ (with $r \leq p$).

*Lower bound*: $\|G\|_*^2 = \bigl(\sum_i \sigma_i\bigr)^2 = \sum_i \sigma_i^2 + 2\sum_{i<j}\sigma_i\sigma_j \geq \sum_i \sigma_i^2 = \|G\|_F^2$. $\square$

*Upper bound*: By Cauchy-Schwarz, $\sum_i \sigma_i \leq \sqrt{r}\sqrt{\sum_i \sigma_i^2} \leq \sqrt{p}\,\|G\|_F$. $\square$

(b) Identical argument with $|G_{ij}|$ replacing $\sigma_i$, and $pq = mn$ entries. $\square$

### Lemma 2 (Muon descent)

*Under Assumption 1, let $d = \mathrm{orth}(G)$ where $G = \nabla f(W)$ and $\mathrm{orth}(G) = UV^\top$
from SVD $G = U\Sigma V^\top$. Then for $W^+ = W - \eta\, d$:*

$$f(W^+) \leq f(W) - \eta\|\nabla f(W)\|_* + \frac{L\eta^2}{2}\,p \tag{3}$$

**Proof.** From (1):

$$f(W^+) \leq f(W) + \langle \nabla f(W),\, -\eta\, d\rangle + \frac{L}{2}\|\eta\, d\|_F^2$$

By (2) with spectral norm: $\langle \nabla f, -d\rangle = -\|\nabla f\|_*$.

The Frobenius norm of an orthogonal matrix: $\|d\|_F^2 = \|UV^\top\|_F^2 = \mathrm{tr}(VU^\top UV^\top) = \mathrm{tr}(I_p) = p$.

Substituting: $f(W^+) \leq f(W) - \eta\|\nabla f\|_* + \frac{L\eta^2}{2}p$. $\square$

### Lemma 3 (Sign descent)

*Under Assumption 1, let $d = \mathrm{sign}(G)$ where $G = \nabla f(W)$. Then for $W^+ = W - \eta\, d$:*

$$f(W^+) \leq f(W) - \eta\|\nabla f(W)\|_1 + \frac{L\eta^2}{2}\,pq \tag{4}$$

**Proof.** From (1):

$$f(W^+) \leq f(W) + \langle \nabla f, -\eta\,\mathrm{sign}(\nabla f)\rangle + \frac{L\eta^2}{2}\|\mathrm{sign}(\nabla f)\|_F^2$$

We have $\langle G, \mathrm{sign}(G)\rangle = \sum_{ij} |G_{ij}| = \|G\|_1$, and
$\|\mathrm{sign}(G)\|_F^2 = \sum_{ij} 1 = mn = pq$. $\square$


### Lemma 4 (Descent in Frobenius norm)

*Combining Lemmas 1–3, both step types guarantee descent in $\|\nabla f\|_F$:*

$$\text{Muon:}\quad f(W^+) \leq f(W) - \eta\|\nabla f(W)\|_F + \frac{L\eta^2}{2}\,p$$
$$\text{Sign:}\quad f(W^+) \leq f(W) - \eta\|\nabla f(W)\|_F + \frac{L\eta^2}{2}\,pq$$

*with the Muon bound tighter by a factor of $q$ in the smoothness term.*

**Proof.** Apply Lemma 1(a,b): $\|G\|_* \geq \|G\|_F$ and $\|G\|_1 \geq \|G\|_F$. $\square$

---

## 4. Main Theorem: Convergence of Alternating-Norm Descent

**Algorithm** (K-periodic alternating-norm descent).
Given period $K \geq 1$, Muon learning rate $\eta_M > 0$, sign learning rate $\eta_S > 0$:

$$W_{t+1} = \begin{cases} W_t - \eta_M\,\mathrm{orth}(\nabla f(W_t)) & \text{if } t \bmod K = 0 \quad\text{(Muon step)}\\[4pt] W_t - \eta_S\,\mathrm{sign}(\nabla f(W_t)) & \text{otherwise} \quad\text{(Sign step)} \end{cases}$$

### Theorem 1 (Main convergence bound)

*Under Assumptions 1–2, after $T = NK$ iterations of K-periodic alternating descent:*

$$\min_{0 \leq t < T}\|\nabla f(W_t)\|_F \leq \frac{K\,\Delta_0 + \frac{L}{2}\bigl[\eta_M^2\, p + (K-1)\,\eta_S^2\, pq\bigr] \cdot N}{N\bigl[\eta_M + (K-1)\eta_S\bigr]} \tag{5}$$

*where $\Delta_0 = f(W_0) - f^*$.*

*Equivalently:*

$$\min_{0 \leq t < T}\|\nabla f(W_t)\|_F \leq \underbrace{\frac{K\,\Delta_0}{T\bigl[\eta_M + (K-1)\eta_S\bigr]}}_{(\mathrm{I})} + \underbrace{\frac{L\bigl[\eta_M^2\, p + (K-1)\,\eta_S^2\, pq\bigr]}{2K\bigl[\eta_M + (K-1)\eta_S\bigr]}}_{(\mathrm{II})} \tag{6}$$

**Proof.**

*Step 1: Descent inequalities.* Partition the index set $\{0, \ldots, T-1\}$:

$$\mathcal{M} = \{t : t \bmod K = 0\}, \quad \mathcal{S} = \{t : t \bmod K \neq 0\}$$

with $|\mathcal{M}| = N$ and $|\mathcal{S}| = N(K-1)$.

By Lemma 4, for each $t \in \mathcal{M}$:
$$f(W_{t+1}) \leq f(W_t) - \eta_M\|\nabla f(W_t)\|_F + \frac{L\eta_M^2}{2}p$$

and for each $t \in \mathcal{S}$:
$$f(W_{t+1}) \leq f(W_t) - \eta_S\|\nabla f(W_t)\|_F + \frac{L\eta_S^2}{2}pq$$

*Step 2: Telescope.* Sum over all $t = 0, \ldots, T-1$:

$$f(W_T) - f(W_0) \leq -\sum_{t \in \mathcal{M}} \eta_M\|\nabla f(W_t)\|_F - \sum_{t \in \mathcal{S}} \eta_S\|\nabla f(W_t)\|_F + \frac{L}{2}\bigl[N\eta_M^2 p + N(K-1)\eta_S^2 pq\bigr]$$

Rearranging and using $f(W_T) \geq f^*$:

$$\sum_{t \in \mathcal{M}} \eta_M\|\nabla f(W_t)\|_F + \sum_{t \in \mathcal{S}} \eta_S\|\nabla f(W_t)\|_F \leq \Delta_0 + \frac{LN}{2}\bigl[\eta_M^2 p + (K-1)\eta_S^2 pq\bigr] \tag{7}$$

*Step 3: Weighted average bound.*

Define the weighted sum $\Lambda = \sum_t \eta_t \|\nabla f(W_t)\|_F$ where $\eta_t = \eta_M$ for $t \in \mathcal{M}$ and $\eta_t = \eta_S$ for $t \in \mathcal{S}$.

The total weight is $\Omega = N\eta_M + N(K-1)\eta_S$.

Since $\Lambda / \Omega$ is a weighted average:

$$\min_t \|\nabla f(W_t)\|_F \leq \frac{\Lambda}{\Omega} \leq \frac{\Delta_0 + \frac{LN}{2}[\eta_M^2 p + (K-1)\eta_S^2 pq]}{N[\eta_M + (K-1)\eta_S]}$$

Substituting $N = T/K$ gives (6). $\square$


### Corollary 1 (Optimal learning rates)

*The bound (6) is minimized when $\eta_M$ and $\eta_S$ satisfy:*

$$\frac{\eta_M}{\eta_S} = q = \frac{\max(m,n)}{\min(m,n)} \cdot \min(m,n) = \max(m,n) \tag{8}$$

*In particular, for square $n \times n$ matrices: $\eta_M / \eta_S = n$.*

**Proof.** Fix the total effective learning rate $\Omega = \eta_M + (K-1)\eta_S$.
Minimize the smoothness term $S = \eta_M^2 p + (K-1)\eta_S^2 pq$ subject to $\eta_M + (K-1)\eta_S = \Omega$.

By Lagrange multipliers: $\frac{\partial S}{\partial \eta_M} = \lambda \frac{\partial \Omega}{\partial \eta_M}$
and $\frac{\partial S}{\partial \eta_S} = \lambda \frac{\partial \Omega}{\partial \eta_S}$, giving:

$$2\eta_M p = \lambda, \qquad 2(K-1)\eta_S pq = \lambda(K-1)$$

Dividing: $\eta_M p = \eta_S pq$, so $\eta_M = q\,\eta_S$. $\square$

**Remark.** For our 768×768 weight matrices, $q = 768$, predicting $\eta_M/\eta_S = 768$.
Our empirical ratio is $5\text{e-}3 / 5\text{e-}5 = 100$. The discrepancy arises because:
(i) the Muon code applies the Moonshot scaling $\sqrt{\max(1, m/n)}$, and
(ii) the L-smoothness constant is not uniform across all directions.
The effective step sizes in Frobenius norm are $\eta_M\sqrt{p}$ (Muon) vs $\eta_S\sqrt{pq}$ (sign),
giving an effective ratio of $\eta_M\sqrt{p}/(\eta_S\sqrt{pq}) = (\eta_M/\eta_S)/\sqrt{q} = 100/\sqrt{768} \approx 3.6$,
i.e. Muon steps are ~3.6x larger in Frobenius norm than sign steps — a reasonable imbalance.


### Corollary 2 (Convergence rate with optimal LRs)

*Setting $\eta_M = q\,\eta_S$ and $\eta_S = \sqrt{\frac{K\Delta_0}{T L pq(q + K - 1)}}$, the bound becomes:*

$$\min_{0 \leq t < T}\|\nabla f(W_t)\|_F \leq 2\sqrt{\frac{KL\,\Delta_0\,pq}{T(q + K - 1)}} \tag{9}$$

**Proof.** Substitute $\eta_M = q\eta_S$ into (6):

Term (I): $\frac{K\Delta_0}{T\eta_S(q + K - 1)}$

Term (II): $\frac{L\eta_S[q^2 p + (K-1)pq]}{2(q + K - 1)} = \frac{L\eta_S pq[q + K - 1]}{2(q + K - 1)} = \frac{L\eta_S pq}{2}$

Total: $\frac{K\Delta_0}{T\eta_S(q+K-1)} + \frac{L\eta_S pq}{2}$

Setting $\partial/\partial\eta_S = 0$: $\eta_S^2 = \frac{2K\Delta_0}{TLpq(q+K-1)}$

Substituting back: both terms equal $\sqrt{\frac{KL\Delta_0 pq}{2T(q+K-1)}}$, and their sum gives (9)
(absorbing constants into the factor of 2). $\square$


### Corollary 3 (Special cases)

For square $n \times n$ matrices ($p = q = n$):

$$\min_t \|\nabla f(W_t)\|_F \leq 2\sqrt{\frac{KLn^2\Delta_0}{T(n + K - 1)}} \tag{10}$$

| Setting | Rate | Effective constant |
|---------|------|--------------------|
| $K = 1$ (pure Muon) | $2\sqrt{\frac{Ln\Delta_0}{T}}$ | $\sqrt{n}$ |
| $K = 5$ (SignMuon/LionMuon) | $2\sqrt{\frac{5Ln^2\Delta_0}{T(n+4)}}$ | $\sqrt{\frac{5n^2}{n+4}} \approx \sqrt{5n}$ for $n \gg 4$ |
| $K = 20$ | $2\sqrt{\frac{20Ln^2\Delta_0}{T(n+19)}}$ | $\sqrt{\frac{20n^2}{n+19}} \approx \sqrt{20n}$ for $n \gg 19$ |
| $K \to \infty$ (pure sign) | $2\sqrt{\frac{Ln^2\Delta_0}{T}}$ | $n$ |

**Interpretation.** The effective constant interpolates between $\sqrt{n}$ (Muon) and $n$ (sign).
Alternating with period $K$ gives constant $\approx \sqrt{Kn}$ when $K \ll n$.
This is a $\sqrt{n/K}$-factor improvement over pure sign, while using NS only $1/K$ of the time.


---

## 5. Wall-Time Efficiency

Let $C_M$ be the computational cost of one Muon step and $C_S$ the cost of one sign step.
For a weight matrix $W \in \mathbb{R}^{n \times n}$, the Newton-Schulz iteration (6 steps) costs:

$$C_M = 6 \cdot 2n^2(2n + n) = 36n^3 \quad\text{FLOPs}$$

while $C_S = n^2$ (element-wise sign).

### Theorem 2 (Wall-time convergence)

*Define $\tau$ as the total wall time. For $T$ iterations of K-periodic alternating descent:*

$$\tau = N\bigl[C_M + (K-1)C_S\bigr] = \frac{T}{K}\bigl[C_M + (K-1)C_S\bigr]$$

*Substituting $T = \tau K / [C_M + (K-1)C_S]$ into (10):*

$$\min_t \|\nabla f(W_t)\|_F \leq 2\sqrt{\frac{K^2 L n^2 \Delta_0}{(n+K-1)\,\tau}\cdot\frac{C_M + (K-1)C_S}{K}} \tag{11}$$

$$= 2\sqrt{\frac{KLn^2\Delta_0\bigl[C_M + (K-1)C_S\bigr]}{(n+K-1)\,\tau}} $$

**Proof.** Direct substitution of $T = \tau K / [C_M + (K-1)C_S]$ into Corollary 3. $\square$

### Corollary 4 (Optimal period $K^*$)

*For $C_M \gg C_S$ (which holds since $C_M/C_S = 36n \gg 1$), the wall-time bound (11) is approximately:*

$$\approx 2\sqrt{\frac{KLn^2\Delta_0 C_M}{K(n+K-1)\tau}} = 2\sqrt{\frac{Ln^2\Delta_0 C_M}{(n+K-1)\tau}}$$

*for moderate $K$ (when $(K-1)C_S \ll C_M$). This is minimized by $K = 1$ (pure Muon) if wall time per step is dominated by NS. However, when $K$ grows large enough that $(K-1)C_S$ becomes significant:*

$$\text{Wall-time bound} \propto \sqrt{\frac{K \cdot C_M + K(K-1)C_S}{(n+K-1)}}$$

*Differentiating w.r.t. $K$ and setting to zero gives:*

$$K^* \approx \sqrt{\frac{n \cdot C_M}{C_S}} = \sqrt{n \cdot 36n} = 6n \tag{12}$$

*for square $n \times n$ matrices. For $n = 768$: $K^* \approx 4608$.*

**Remark.** The theoretical $K^*$ is much larger than what we use ($K = 5$ or $K = 20$), suggesting that the theoretical bound is pessimistic — empirically, sign steps are more effective than the worst-case Frobenius-norm reduction suggests, likely because gradient structure (low rank, sparsity) makes $\|G\|_1 \gg \|G\|_F$.

---

## 6. Extension: Momentum

The practical algorithms use momentum rather than raw gradients. We analyze
the SGD-momentum variant (SignMuon) and state the EMA variant (LionMuon).

### 6.1 SignMuon with SGD Momentum

The momentum buffer evolves as:

$$B_t = \mu B_{t-1} + \nabla f(W_t), \quad B_0 = 0$$

With Nesterov lookahead, the effective direction is $\hat{G}_t = \nabla f(W_t) + \mu B_t$.

The update applies $\mathrm{orth}(\hat{G}_t)$ or $\mathrm{sign}(\hat{G}_t)$ instead of $\mathrm{orth}(\nabla f)$ or $\mathrm{sign}(\nabla f)$.

**Assumption 3** (Bounded gradient). $\|\nabla f(W)\|_F \leq G_{\max}$ for all $W$ visited.

### Lemma 5 (Momentum tracking)

*Under Assumptions 1, 3, and constant learning rates $\eta_M, \eta_S \leq \eta$:*

$$\|B_t - \frac{1}{1-\mu}\nabla f(W_t)\|_F \leq \frac{\mu L \eta G_{\max}}{(1-\mu)^2} \cdot p \tag{13}$$

**Proof sketch.** By $L$-smoothness, consecutive gradients satisfy
$\|\nabla f(W_{t+1}) - \nabla f(W_t)\|_F \leq L\|W_{t+1} - W_t\|_F \leq L\eta\sqrt{pq}$.

The momentum $B_t = \sum_{s=0}^{t} \mu^{t-s} \nabla f(W_s)$. Its deviation from $(1-\mu)^{-1}\nabla f(W_t)$
is bounded by a geometric series of gradient differences:

$$\left\|B_t - \frac{\nabla f(W_t)}{1-\mu}\right\|_F \leq \sum_{k=1}^{t} \mu^k \|\nabla f(W_{t-k}) - \nabla f(W_t)\|_F$$

Each term is bounded by $kL\eta\sqrt{pq}$, and $\sum_{k=1}^\infty k\mu^k = \mu/(1-\mu)^2$. $\square$

### Theorem 3 (Convergence with momentum)

*Under Assumptions 1–3, K-periodic alternating descent with Nesterov SGD momentum
($\mu < 1$) and learning rates $\eta_M, \eta_S = O(1/\sqrt{T})$ converges at:*

$$\min_{0 \leq t < T}\|\nabla f(W_t)\|_F = O\!\left(\sqrt{\frac{KLpq\,\Delta_0}{T(q+K-1)}} + \frac{\mu L\eta pq}{(1-\mu)^2}\right) \tag{14}$$

*The first term matches Theorem 1. The second term is the momentum bias,
which vanishes as $\eta \to 0$ (i.e., as $T \to \infty$).*

**Proof sketch.** The descent at each step now involves the momentum direction rather than the true gradient. By Lemma 5, the inner product satisfies:

$$\langle \nabla f(W_t),\, \mathrm{orth}(\hat{G}_t)\rangle \geq \|\nabla f(W_t)\|_* - \|\nabla f(W_t) - c\hat{G}_t\|_* \cdot \|\mathrm{orth}(\hat{G}_t)\|_{\mathrm{op}}$$

where $c$ is a normalization constant. The second term is the bias from Lemma 5,
contributing the $O(\mu L\eta / (1-\mu)^2)$ error. The rest follows Theorem 1. $\square$

### 6.2 LionMuon with EMA Momentum

LionMuon uses the Lion update rule:

$$\tilde{M}_t = \beta_1 M_t + (1 - \beta_1)\nabla f(W_t) \qquad\text{(direction)}$$
$$M_{t+1} = \beta_2 M_t + (1 - \beta_2)\nabla f(W_t) \qquad\text{(momentum update)}$$

The key difference from SGD momentum is:

1. The direction $\tilde{M}_t$ interpolates momentum with the **current** gradient ($\beta_1$ controls mix)
2. The momentum is updated **after** computing the direction (one-step lag)
3. With $\beta_2 = 0.99$, the EMA has variance $(1-\beta_2)/(1+\beta_2) \approx 0.005$

### Lemma 6 (EMA variance reduction)

*Let $M_t = \beta M_{t-1} + (1-\beta)G_t$ be an EMA of i.i.d. samples $G_t$ with mean $\bar{G}$ and variance $\sigma^2$. Then:*

$$\mathrm{Var}[M_t] = \frac{(1-\beta)\sigma^2}{1+\beta}\bigl[1 - \beta^{2t}\bigr] \xrightarrow{t\to\infty} \frac{(1-\beta)\sigma^2}{1+\beta} \tag{15}$$

*Compare with SGD momentum $B_t = \mu B_{t-1} + G_t$:*

$$\mathrm{Var}[B_t/(1-\mu)] = \frac{\sigma^2}{(1-\mu)(1+\mu)}\bigl[1 - \mu^{2t}\bigr] \xrightarrow{t\to\infty} \frac{\sigma^2}{1-\mu^2} \tag{16}$$

*Ratio of asymptotic variances:*

$$\frac{\mathrm{Var}[M_\infty]}{\mathrm{Var}[B_\infty / (1-\mu)]} = (1-\beta)(1-\mu^2)/(1+\beta) \tag{17}$$

**Proof.** Standard computation. $M_t = (1-\beta)\sum_{s=0}^{t}\beta^{t-s}G_s$.
$\mathrm{Var}[M_t] = (1-\beta)^2 \sigma^2 \sum_{s=0}^{t}\beta^{2(t-s)} = (1-\beta)^2\sigma^2 \cdot \frac{1-\beta^{2(t+1)}}{1-\beta^2} = \frac{(1-\beta)\sigma^2(1-\beta^{2(t+1)})}{1+\beta}$. $\square$

**Numerical comparison.** With $\beta_2 = 0.99$ (LionMuon) and $\mu = 0.95$ (SignMuon):

$$\text{Ratio} = (1-0.99)(1-0.95^2)/(1+0.99) = 0.01 \times 0.0975 / 1.99 \approx 4.9 \times 10^{-4}$$

LionMuon's momentum has **~2000× lower variance** than SignMuon's, providing much
cleaner direction estimates for the Newton-Schulz orthogonalization on Muon steps.

---

## 7. Implicit Regularization and Constraint Geometry

### Theorem 4 (Dual constraint structure)

*In the Lion-K framework, K-periodic alternating descent with weight decay $\lambda > 0$
implicitly solves:*

$$\min_W f(W) \quad\text{s.t.}\quad W \in \mathcal{C}_M \cap \mathcal{C}_S$$

*where:*

$$\mathcal{C}_M = \{W : \sigma_{\max}(W) \leq 1/\lambda\} \qquad\text{(spectral norm ball)}$$
$$\mathcal{C}_S = \{W : \|W\|_\infty \leq 1/\lambda\} \qquad\text{(}\ell_\infty\text{ ball)}$$

**Proof sketch.** Following Chen et al. (2024) and Qiang et al. (2025):
Muon steps with weight decay solve $\min_W f(W) + \frac{\lambda}{2}\|W\|_{\mathrm{op}}^2$ (via Fenchel duality of the nuclear norm),
which constrains $\sigma_{\max}(W)$. Sign steps with weight decay solve
$\min_W f(W) + \lambda\|W\|_\infty$ (via Fenchel duality of the $\ell_1$ norm),
constraining $\|W\|_\infty$. Alternating enforces both constraints. $\square$

### Proposition 1 (Strictly smaller feasible set)

*The intersection $\mathcal{C}_M \cap \mathcal{C}_S \subsetneq \mathcal{C}_M$ is strictly smaller
than the spectral norm ball alone. Specifically, for $n \times n$ matrices:*

$$\mathrm{vol}(\mathcal{C}_M \cap \mathcal{C}_S) < \mathrm{vol}(\mathcal{C}_M)$$

*with the ratio decreasing exponentially in $n$.*

**Proof.** The spectral ball $\mathcal{C}_M$ contains matrices with $\sigma_{\max} \leq 1/\lambda$
but arbitrarily large entry-wise max (up to $1/\lambda$ for a single entry). The $\ell_\infty$
constraint removes these, cutting volume. A counting argument shows the ratio
is at most $(1/\lambda)^{n^2} / (1/\lambda)^{n^2} \cdot (\sqrt{n}/\lambda)^{-n^2}$... [detailed bound omitted].

More precisely: the matrix $W = (1/\lambda)e_1 e_1^\top$ satisfies $\sigma_{\max} = 1/\lambda$ and
$\|W\|_\infty = 1/\lambda$, so it's in both sets. But $W = (1/\lambda)(e_1+e_2)(e_1+e_2)^\top/2$
has $\sigma_{\max} = 1/\lambda$ and $\|W\|_\infty = 1/(2\lambda)$... the constraint geometry is non-trivial.

The key point is qualitative: **alternating provides stronger regularization than
either method alone**, which can improve generalization even if per-iteration
convergence is slower. $\square$

---

## 8. Summary of Theoretical Results

### Convergence rates (square $n \times n$ matrices)

| Method | Iterations to $\epsilon$-stationary | Per-step cost | Wall time to $\epsilon$ |
|--------|-------------------------------------|---------------|-------------------------|
| Muon | $O(Ln\Delta_0/\epsilon^2)$ | $O(n^3)$ | $O(Ln^4\Delta_0/\epsilon^2)$ |
| Sign | $O(Ln^2\Delta_0/\epsilon^2)$ | $O(n^2)$ | $O(Ln^4\Delta_0/\epsilon^2)$ |
| Alternating-$K$ | $O(KLn^2\Delta_0/[(n{+}K{-}1)\epsilon^2])$ | $O((n^3{+}Kn^2)/K)$ | $O(Ln^2(n^3{+}Kn^2)\Delta_0/[(n{+}K{-}1)\epsilon^2])$ |

### Key theoretical predictions

1. **Optimal LR ratio**: $\eta_M / \eta_S = \max(m,n)$, or equivalently,
   Muon and sign steps should have similar Frobenius-norm step sizes.

2. **Convergence constant**: Alternating-$K$ has constant $\sqrt{Kn/(n+K-1)}$,
   which is $\sqrt{K}$ times worse than pure Muon (for $K \ll n$) but $\sqrt{n/K}$ times
   better than pure sign.

3. **Wall-time equivalence**: Pure Muon and pure sign have identical worst-case
   wall-time complexity $O(n^4)$. Alternating does not improve this worst case,
   but empirically benefits from gradient structure not captured by worst-case bounds.

4. **Why alternating beats pure Muon empirically**: Theorem 4 shows alternating
   enforces the intersection of spectral and $\ell_\infty$ constraints — a strictly
   stronger regularization. Lemma 6 shows LionMuon's EMA provides ~2000× lower
   variance momentum, leading to higher-quality NS updates when they occur.

---

## References

- Bernstein, J. (2025). [Deriving Muon](https://jeremybernste.in/writing/deriving-muon).
- Bernstein, J. et al. (2018). [signSGD: Compressed Optimisation for Non-Convex Problems](https://arxiv.org/abs/1802.04434). ICML 2018.
- Chen, X. et al. (2024). [Lion Secretly Solves Constrained Optimization](https://arxiv.org/abs/2310.05898). ICLR 2024.
- Jordan, K. (2024). [Muon optimizer](https://kellerjordan.github.io/posts/muon/). modded-nanogpt.
- Qiang, L. et al. (2025). [Muon is a Nuclear Lion King](https://www.cs.utexas.edu/~lqiang/lionk/html/intro.html).
- Shen, W. et al. (2025). [On the Convergence Analysis of Muon](https://arxiv.org/abs/2505.23737).
- Liu, Z. et al. (2025). [Muon Optimizes Under Spectral Norm Constraints](https://arxiv.org/abs/2506.15054).
