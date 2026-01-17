import torch
from torch.optim.optimizer import Optimizer


def _matrix_power(matrix, power):
    # Keep on GPU for speed - use eigendecomposition for symmetric matrices
    # Since precond = G @ G^T is symmetric positive semi-definite
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    eigvals = eigvals.clamp(min=1e-10)  # numerical stability
    return eigvecs @ (eigvals.pow(power).diag()) @ eigvecs.T


class Shampoo(Optimizer):
    """
    Implements Shampoo algorithm.
    
    Based on: https://arxiv.org/abs/1802.09568
    """

    def __init__(self, params, lr=1e-1, momentum=0, weight_decay=0, epsilon=1e-4, 
                 update_freq=1, power=0.25, max_precond_dim=10000):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, 
                       epsilon=epsilon, update_freq=update_freq, power=power,
                       max_precond_dim=max_precond_dim)
        super(Shampoo, self).__init__(params, defaults)

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
                order = grad.ndimension()
                original_size = grad.size()
                state = self.state[p]
                momentum = group["momentum"]
                weight_decay = group["weight_decay"]
                power = group["power"]

                if len(state) == 0:
                    state["step"] = 0
                    if momentum > 0:
                        state["momentum_buffer"] = grad.clone()
                    for dim_id, dim in enumerate(grad.size()):
                        # skip large dimensions to save memory
                        if dim > group["max_precond_dim"]:
                            state[f"precond_{dim_id}"] = None
                            state[f"inv_precond_{dim_id}"] = None
                        else:
                            state[f"precond_{dim_id}"] = group["epsilon"] * torch.eye(dim, device=grad.device, dtype=grad.dtype)
                            state[f"inv_precond_{dim_id}"] = torch.zeros(dim, dim, device=grad.device, dtype=grad.dtype)

                if momentum > 0:
                    # grad = (1 - moment) * grad(t) + moment * grad(t-1)
                    # and grad(-1) = grad(0)
                    grad = grad.mul(1 - momentum).add_(state["momentum_buffer"], alpha=momentum)

                if weight_decay > 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # See Algorithm 2 for detail
                for dim_id, dim in enumerate(grad.size()):
                    precond = state[f"precond_{dim_id}"]
                    inv_precond = state[f"inv_precond_{dim_id}"]

                    # skip large dimensions
                    if precond is None:
                        continue

                    # mat_{dim_id}(grad)
                    grad = grad.transpose_(0, dim_id).contiguous()
                    transposed_size = grad.size()
                    grad = grad.view(dim, -1)

                    grad_t = grad.t()
                    precond.add_(grad @ grad_t)
                    if state["step"] % group["update_freq"] == 0:
                        inv_precond.copy_(_matrix_power(precond, -power))

                    if dim_id == order - 1:
                        grad = grad_t @ inv_precond
                        grad = grad.view(original_size)
                    else:
                        grad = inv_precond @ grad
                        grad = grad.view(transposed_size)

                state["step"] += 1
                state["momentum_buffer"] = grad
                p.data.add_(grad, alpha=-group["lr"])

        return loss
