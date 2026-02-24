# SignMuon

**Input:** learning rate $\eta$, sign learning rate $\eta_s$, momentum $\mu$, $K = 5$

**Initialize:** $B \leftarrow 0$, $t \leftarrow 0$

**For** each step:

$$B \leftarrow \mu B + G, \quad \hat{G} \leftarrow G + \mu B$$

**If** $t \mod K = 0$:

$$W \leftarrow W - \eta \cdot 0.2 \sqrt{\max(m, n)} \cdot \text{NewtonSchulz}(\hat{G})$$

**Else:**

$$W \leftarrow W - \eta_s \cdot \text{sign}(\hat{G})$$

$$t \leftarrow t + 1$$

---

# LionMuon

**Input:** learning rate $\eta$, Lion learning rate $\eta_\ell$, $\beta_1$, $\beta_2$, $K = 5$

**Initialize:** $M \leftarrow 0$, $t \leftarrow 0$

**For** each step:

$$\hat{G} \leftarrow \beta_1 M + (1 - \beta_1) G$$

**If** $t \mod K = 0$:

$$W \leftarrow W - \eta \cdot 0.2 \sqrt{\max(m, n)} \cdot \text{NewtonSchulz}(\hat{G})$$

**Else:**

$$W \leftarrow W - \eta_\ell \cdot \text{sign}(\hat{G})$$

$$M \leftarrow \beta_2 M + (1 - \beta_2) G, \quad t \leftarrow t + 1$$
