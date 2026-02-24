import math
from typing import Iterable, Optional, Tuple

import torch


class AdamSania(torch.optim.Optimizer):
    """Adam-SANIA optimizer.

    Same as Adam, except the second moment is used *without* sqrt in the
    denominator. Implements update:

        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)

    Define B_t = v_hat + eps (elementwise). Step size factor per element:
        lambda_t = 1 - sqrt(1 - v_hat)    if v_hat <= 1
                    1                      otherwise

    Update:
        w <- w - lr * lambda_t * (m_hat / B_t)

    Weight decay (AdamW style) is applied decoupled from gradient.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Decoupled weight decay (AdamW style)
                if wd != 0:
                    p.mul_(1 - lr * wd)

                # Moments
                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                # Bias correction
                bias_c1 = 1 - beta1 ** step
                bias_c2 = 1 - beta2 ** step
                m_hat = exp_avg / bias_c1
                v_hat = exp_avg_sq / bias_c2

                # SANIA step-size lambda_t per element
                # lambda = (v_hat <= 1) ? (1 - sqrt(1 - v_hat)) : 1
                # ensure numeric stability for values slightly >1 due to fp error
                lambda_t = torch.where(
                    v_hat <= 1,
                    1 - torch.sqrt(torch.clamp(1 - v_hat, min=0.0)),
                    torch.ones_like(v_hat),
                )

                denom = v_hat + eps
                update = lambda_t * (m_hat / denom)
                p.add_(update, alpha=-lr)

        return loss
