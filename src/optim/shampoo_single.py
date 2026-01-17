import torch
from torch.optim.optimizer import Optimizer


def _matrix_power(matrix, power):
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    eigvals = eigvals.clamp(min=1e-10)
    return eigvecs @ (eigvals.pow(power).diag()) @ eigvecs.T


class ShampooSingle(Optimizer):
    """One-sided Shampoo: left = L^{-power} G, right = G R^{-power}"""

    def __init__(self, params, lr=1e-1, momentum=0, weight_decay=0, epsilon=1e-4, 
                 update_freq=1, power=0.5, side='left', max_precond_dim=10000):
        assert side in ('left', 'right'), "side must be 'left' or 'right'"
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, 
                       epsilon=epsilon, update_freq=update_freq, power=power, side=side,
                       max_precond_dim=max_precond_dim)
        super(ShampooSingle, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                original_size = grad.size()
                state = self.state[p]
                momentum = group["momentum"]
                weight_decay = group["weight_decay"]
                power = group["power"]
                side = group["side"]

                # Reshape to 2D
                if grad.ndim == 1:
                    grad = grad.view(-1, 1)
                    reshaped = True
                elif grad.ndim > 2:
                    grad = grad.view(grad.size(0), -1)
                    reshaped = True
                else:
                    reshaped = False

                m, n = grad.size()

                if len(state) == 0:
                    state["step"] = 0
                    if momentum > 0:
                        state["momentum_buffer"] = grad.clone()
                    precond_dim = m if side == 'left' else n
                    if precond_dim > group["max_precond_dim"]:
                        state["precond"] = None
                    elif side == 'left':
                        state["precond"] = group["epsilon"] * torch.eye(m, device=grad.device, dtype=grad.dtype)
                    else:  # right
                        state["precond"] = group["epsilon"] * torch.eye(n, device=grad.device, dtype=grad.dtype)
                    state["inv_precond"] = None

                if momentum > 0:
                    grad = grad.mul(1 - momentum).add_(state["momentum_buffer"], alpha=momentum)

                if weight_decay > 0:
                    grad_for_wd = p.data.view(m, n) if reshaped else p.data
                    grad = grad.add(grad_for_wd, alpha=weight_decay)

                precond = state["precond"]

                # skip if dimension too large
                if precond is not None:
                    if side == 'left':
                        precond.add_(grad @ grad.t())
                    else:
                        precond.add_(grad.t() @ grad)

                    if state["step"] % group["update_freq"] == 0:
                        state["inv_precond"] = _matrix_power(precond, -power)

                    inv_precond = state["inv_precond"]
                    if inv_precond is not None:
                        if side == 'left':
                            grad = inv_precond @ grad
                        else:
                            grad = grad @ inv_precond

                state["step"] += 1
                if momentum > 0:
                    state["momentum_buffer"] = grad.clone()

                # Reshape back to original size
                grad = grad.view(original_size)
                p.data.add_(grad, alpha=-group["lr"])

        return loss
