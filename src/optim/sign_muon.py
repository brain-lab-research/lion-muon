"""
SignMuon: Alternating cheap and full Muon optimizer.

Motivation: Muon (Newton-Schulz orthogonalization) is expensive per iteration but
gives high-quality spectral-norm updates. We can save wall time by doing full Muon
every K steps and using a cheaper update in between.

Cheap update modes:
- "sign": sign(momentum) — infinity-norm LMO, cheapest but lowest quality
- "norm": momentum / ‖momentum‖_F — Frobenius-norm LMO, preserves gradient magnitudes
- "cheap_ns": NS with fewer steps (e.g. 1-2) — still spectral, much better than sign
- "cached": reuse last full Muon direction — zero compute, full spectral quality

References:
- Muon: MomentUm Orthogonalized by Newton-Schulz (Keller Jordan, modded-nanogpt)
- SignSGD / Signum: Bernstein et al., "signSGD: Compressed Optimisation for Non-Convex Problems"
"""

import os

import torch
import torch.distributed as dist

from .muon import zeropower_via_newtonschulz5
from .schedule import cos_inf_schedule, cosine_wsd_decay_schedule, wsd_schedule


class SignMuon(torch.optim.Optimizer):
    """
    SignMuon alternates between a cheap update and full Muon for 2D parameters.
    For non-2D parameters (embeddings, biases, layernorms), it always uses AdamW.

    Arguments:
        muon_params: Parameters to optimize.
        lr: Learning rate for Muon steps.
        cheap_lr: Learning rate for cheap steps. If None, defaults to lr.
        momentum: Momentum coefficient (shared between phases).
        nesterov: Use Nesterov momentum.
        ns_steps: Newton-Schulz steps for full Muon.
        muon_every_k: Full Muon every K steps, cheap update for the rest.
        cheap_mode: "sign" | "norm" | "cheap_ns" | "cached"
        cheap_ns_steps: NS steps for cheap_ns mode (default 2).
        sign_scaling: "muon" (frob+aspect), "frob" (frob only), "none" (raw sign).
        adamw_params: Parameters to always optimize with AdamW.
        adamw_lr, adamw_betas, adamw_eps, adamw_wd: AdamW hyperparameters.
    """

    def __init__(
        self,
        muon_params,
        lr=0.02,
        cheap_lr=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=6,
        muon_every_k=5,
        cheap_mode="norm",
        cheap_ns_steps=2,
        sign_scaling="muon",
        weight_decay=0.0,
        adamw_params=None,
        adamw_lr=3e-4,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        adamw_wd=0,
    ):
        if cheap_lr is None:
            if cheap_mode in ("sign", "norm"):
                cheap_lr = lr * 0.25  # sign/norm need ~4x smaller LR (empirical)
            else:
                cheap_lr = lr

        defaults = dict(
            lr=lr,
            cheap_lr=cheap_lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            muon_every_k=muon_every_k,
            cheap_mode=cheap_mode,
            cheap_ns_steps=cheap_ns_steps,
            sign_scaling=sign_scaling,
            weight_decay=weight_decay,
            adamw_lr=adamw_lr,
            adamw_lr_ratio=adamw_lr / lr if lr > 0 else 1.0,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)

        for p in muon_params:
            if p.ndim >= 2 and p.size(0) < 10000:
                self.state[p]["use_muon"] = True
            else:
                self.state[p]["use_muon"] = False
        for p in adamw_params:
            self.state[p]["use_muon"] = False

        self._step_count = 0

        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])
        else:
            self.world_size = 1
            self.rank = 0

    def _is_muon_step(self):
        k = self.param_groups[0]["muon_every_k"]
        if k <= 1:
            return True
        return (self._step_count % k) == 0

    def _muon_update(self, g, ns_steps):
        """Muon update with Moonshot RMS-matching scaling."""
        g = zeropower_via_newtonschulz5(g, steps=ns_steps)
        g *= 0.2 * max(g.size(0), g.size(1)) ** 0.5
        return g

    def _cheap_update(self, g, mode, cheap_ns_steps, sign_scaling="muon", cached_dir=None):
        """Cheap update: sign, normalized, reduced NS steps, or cached direction."""
        if mode == "sign":
            update = g.sign()
            rows, cols = g.size(0), g.size(1) if g.ndim >= 2 else 1
            if sign_scaling == "muon":
                update /= max(rows, cols) ** 0.5
                update *= max(1, rows / cols) ** 0.5
            elif sign_scaling == "frob":
                update /= max(rows, cols) ** 0.5
            # sign_scaling == "none": raw sign, no normalization
            return update
        elif mode == "norm":
            # Frobenius-norm LMO: normalize to unit Frobenius norm,
            # then scale to match Muon's output norm ≈ sqrt(min(rows, cols))
            nrm = g.norm() + 1e-7
            update = g / nrm
            update *= min(g.size(0), g.size(1)) ** 0.5
            update *= max(1, g.size(0) / g.size(1)) ** 0.5
            return update
        elif mode == "cheap_ns":
            return self._muon_update(g, ns_steps=cheap_ns_steps)
        elif mode == "cached":
            if cached_dir is not None:
                return cached_dir
            # First step before any Muon — fall back to norm
            nrm = g.norm() + 1e-7
            update = g / nrm
            update *= min(g.size(0), g.size(1)) ** 0.5
            update *= max(1, g.size(0) / g.size(1)) ** 0.5
            return update
        else:
            raise ValueError(f"Unknown cheap_mode: {mode}")

    def step(self):
        is_muon = self._is_muon_step()
        self._step_count += 1

        for group in self.param_groups:

            ############################
            #   Muon / cheap (2D)      #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            momentum = group["momentum"]
            cheap_mode = group["cheap_mode"]
            cheap_ns_steps = group["cheap_ns_steps"]

            if is_muon:
                lr = group["lr"]
            else:
                lr = group["cheap_lr"]

            # Distributed: compute updates, flatten, all-reduce
            total_params = sum(p.numel() for p in params)
            updates_flat = torch.zeros(
                total_params, device="cuda", dtype=torch.bfloat16
            )
            curr_idx = 0
            for i, p in enumerate(params):
                if i % self.world_size == self.rank:
                    g = p.grad
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if group["nesterov"]:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf.clone()

                    if is_muon:
                        g = self._muon_update(g, ns_steps=group["ns_steps"])
                        state["cached_direction"] = g.clone()
                    else:
                        cached_dir = state.get("cached_direction", None)
                        g = self._cheap_update(g, cheap_mode, cheap_ns_steps, group["sign_scaling"], cached_dir)

                    updates_flat[curr_idx : curr_idx + p.numel()] = g.flatten()
                curr_idx += p.numel()

            if self.world_size > 1:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr_idx = 0
            wd = group["weight_decay"]
            for p in params:
                g = (
                    updates_flat[curr_idx : curr_idx + p.numel()]
                    .view_as(p.data)
                    .type_as(p.data)
                )
                if wd > 0:
                    p.data.mul_(1 - lr * wd)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group["adamw_lr_ratio"] * group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["adamw_wd"]

            for p in params:
                g = p.grad
                assert g is not None
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)


class SignMuonScheduler:
    """
    Scheduler for SignMuon optimizer, handling both muon-phase LR and adamw LR.
    Similar to CombinedScheduler from muon.py.
    """

    def __init__(self, optimizer, cfg, muon_lr_key="lr", adamw_lr_key="adamw_lr"):
        self.schedulers = []
        def _make_cos_scheduler(opt, lr):
            warmup = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=1e-2, total_iters=cfg.warmup_steps
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=cfg.iterations - cfg.warmup_steps, eta_min=0
            )
            return torch.optim.lr_scheduler.SequentialLR(
                opt, [warmup, cosine], milestones=[cfg.warmup_steps]
            )

        scheduler_map = {
            "cos": _make_cos_scheduler,
            "cos_inf": lambda opt, lr: torch.optim.lr_scheduler.LambdaLR(
                opt,
                cos_inf_schedule(
                    n_iterations=cfg.iterations,
                    n_warmup=cfg.warmup_steps,
                    n_inf=cfg.cos_inf_steps,
                    div_factor=1e2,
                    final_div_factor=0.1,
                ),
            ),
            "wsd": lambda opt, lr: torch.optim.lr_scheduler.LambdaLR(
                opt,
                wsd_schedule(
                    n_iterations=cfg.iterations,
                    n_warmup=cfg.warmup_steps,
                    fract_decay=cfg.wsd_fract_decay,
                    init_div_factor=1e2,
                    final_lr_factor=cfg.wsd_final_lr_scale,
                    decay_type=cfg.decay_type,
                ),
            ),
            "cos_wsd": lambda opt, lr: torch.optim.lr_scheduler.LambdaLR(
                opt,
                cosine_wsd_decay_schedule(
                    n_iterations=cfg.iterations,
                    n_warmup=cfg.warmup_steps,
                    anneal_end_factor=0.15,
                    fract_decay=cfg.wsd_fract_decay,
                    init_div_factor=1e2,
                    final_lr_factor=0.1,
                    decay_type=cfg.decay_type,
                ),
            ),
        }

        for group in optimizer.param_groups:
            lr_key = muon_lr_key if muon_lr_key in group else adamw_lr_key
            if lr_key in group:
                scheduler_cls = scheduler_map.get(cfg.scheduler, None)
                if scheduler_cls:
                    scheduler = scheduler_cls(
                        optimizer, group.get(lr_key, getattr(cfg, lr_key.lower()))
                    )
                    self.schedulers.append(scheduler)

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self):
        state_dict = {}
        for i, scheduler in enumerate(self.schedulers):
            state_dict[f"scheduler_{i}"] = scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        for i, scheduler in enumerate(self.schedulers):
            scheduler.load_state_dict(state_dict[f"scheduler_{i}"])
