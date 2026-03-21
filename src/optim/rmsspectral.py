"""
RMSspectral: Four ways to combine spectral LMO (Muon-style) with Adam-style V_t.

Variants:
  1) "post"       — LMO(M_t), then divide by sqrt(V̂_t).           (Clarifying Shampoo style)
  2) "pre"        — LMO(M_t / sqrt(V̂_t)).                         (no paper, "шиза")
  3) "post_orth"  — LMO(M_t), V_t from LMO(g_t)^2, divide.       (AdaMuon style)
  4) "split"      — M_t/√(√V̂+ε) → LMO → /√(√V̂+ε)                (Preconditioned Norms / RMSspectral)
  5) "pre_ema"    — EMA(g_t / sqrt(V̂_t)) → LMO                  (normalize then average)

Only bc2 correction on V_t. No bc1 — NS normalizes spectral norm, so scalar on input is irrelevant.
All variants use AdamW fallback for 1D parameters.
"""

import math

import torch

from .muon import zeropower_via_newtonschulz5


class RMSspectral(torch.optim.Optimizer):

    def __init__(
        self,
        param_groups,
        lr=1e-3,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.0,
        ns_steps=5,
        variant="post",
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):
        assert variant in ("post", "pre", "post_orth", "split", "pre_ema"), (
            f"Unknown variant: {variant}"
        )
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            variant=variant,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        super().__init__(param_groups, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["use_spectral"] = p.ndim >= 2 and p.size(0) < 10000

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:

            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            variant = group["variant"]
            beta1_adamw, beta2_adamw = group["adamw_betas"]
            eps_adamw = group["adamw_eps"]

            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1
            step = group["step"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                # ---- 1D params: AdamW fallback ----
                if not state.get("use_spectral", False):
                    if "adamw_m" not in state:
                        state["adamw_m"] = torch.zeros_like(g)
                        state["adamw_v"] = torch.zeros_like(g)
                    am = state["adamw_m"]
                    av = state["adamw_v"]
                    am.lerp_(g, 1 - beta1_adamw)
                    av.lerp_(g.square(), 1 - beta2_adamw)
                    bc1 = 1 - beta1_adamw ** step
                    bc2 = 1 - beta2_adamw ** step
                    step_size = lr * math.sqrt(bc2) / bc1
                    p.data.mul_(1 - lr * weight_decay)
                    p.data.addcdiv_(am, av.sqrt().add_(eps_adamw), value=-step_size)
                    continue

                # ---- 2D params: spectral + adaptive ----
                if "m" not in state:
                    state["m"] = torch.zeros_like(g)
                    state["v"] = torch.zeros_like(g)

                m = state["m"]
                v = state["v"]
                if variant != "pre_ema":
                    m.lerp_(g, 1 - beta1)

                # Only bc2 for V_t. bc1 irrelevant — NS normalizes spectral norm.
                bc2 = 1 - beta2 ** step

                if variant == "post":
                    v.lerp_(g.square(), 1 - beta2)
                    o_t = zeropower_via_newtonschulz5(m.bfloat16(), steps=ns_steps).to(g.dtype)
                    update = o_t / ((v / bc2).sqrt() + eps)

                elif variant == "pre":
                    v.lerp_(g.square(), 1 - beta2)
                    n_t = m / ((v / bc2).sqrt() + eps)
                    update = zeropower_via_newtonschulz5(n_t.bfloat16(), steps=ns_steps).to(g.dtype)

                elif variant == "post_orth":
                    o_t = zeropower_via_newtonschulz5(m.bfloat16(), steps=ns_steps).to(g.dtype)
                    v.lerp_(o_t.square(), 1 - beta2)
                    update = o_t / ((v / bc2).sqrt() + eps)

                elif variant == "split":
                    v.lerp_(g.square(), 1 - beta2)
                    denom = (v / bc2 + eps).sqrt().sqrt()
                    n_t = m / denom
                    n_prime = zeropower_via_newtonschulz5(n_t.bfloat16(), steps=ns_steps).to(g.dtype)
                    update = n_prime / denom

                elif variant == "pre_ema":
                    v.lerp_(g.square(), 1 - beta2)
                    n_t = g / ((v / bc2).sqrt() + eps)  # normalize grad first
                    m.lerp_(n_t, 1 - beta1)             # then EMA of normalized
                    update = zeropower_via_newtonschulz5(m.bfloat16(), steps=ns_steps).to(g.dtype)

                # Rectangular matrix scaling (same as old Muon)
                update *= max(1, p.size(0) / p.size(1)) ** 0.5

                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(update, alpha=-lr)
