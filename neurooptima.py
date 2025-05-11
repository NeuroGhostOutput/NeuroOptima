import torch
from torch.optim import Optimizer
import math

class NeuroOptima(Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.0,
                 sam_rho=0.05, lookahead_k=5, lookahead_alpha=0.5,
                 betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, weight_decay=weight_decay,
                        sam_rho=sam_rho, lookahead_k=lookahead_k,
                        lookahead_alpha=lookahead_alpha,
                        betas=betas, eps=eps)
        super(NeuroOptima, self).__init__(params, defaults)
        self._step = 0
        self._slow_params = []

        # Initialize slow weights for Lookahead
        for group in self.param_groups:
            slow_group = []
            for p in group['params']:
                if p.requires_grad:
                    slow_group.append(p.data.clone())
                else:
                    slow_group.append(None)
            self._slow_params.append(slow_group)

    @torch.no_grad()
    def step(self, closure):
        loss = closure()
        loss.backward()

        for group_idx, group in enumerate(self.param_groups):
            lr = group['lr']
            weight_decay = group['weight_decay']
            sam_rho = group['sam_rho']
            beta1, beta2 = group['betas']
            eps = group['eps']

            for p_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['exp_avg_diff'] = torch.zeros_like(p.data)
                    state['prev_grad'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg_diff, prev_grad = state['exp_avg_diff'], state['prev_grad']

                state['step'] += 1

                # Adan updates
                grad_diff = grad - prev_grad
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_diff.mul_(beta2).add_(grad_diff, alpha=1 - beta2)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(eps)
                update = (exp_avg + beta2 * exp_avg_diff) / denom

                # Lion update (sign-based)
                p.data.add_(update.sign(), alpha=-lr)

                # SAM perturbation
                if sam_rho > 0:
                    grad_norm = torch.norm(grad)
                    if grad_norm != 0:
                        scale = sam_rho / (grad_norm + eps)
                        p.data.add_(grad, alpha=scale)

                # Save current grad for next step
                prev_grad.copy_(grad)

        self._step += 1

        # Lookahead synchronization
        for group_idx, group in enumerate(self.param_groups):
            if self._step % group['lookahead_k'] == 0:
                for p_idx, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    slow = self._slow_params[group_idx][p_idx]
                    if slow is None:
                        continue
                    slow.add_(p.data - slow, alpha=group['lookahead_alpha'])
                    p.data.copy_(slow)

        return loss
