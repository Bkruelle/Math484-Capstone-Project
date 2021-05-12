#MATH CAPSTONE NOTE: This algorithm includes HAdam in addition to AdamW and Demon. Letting k=2 (default) is the same as not using HAdam, and is what was used for all tests produced for the project.


import torch
from torch.optim import Optimizer

class HAdamWDemon(Optimizer):
    """Implements HAdamWDemon algorithm (a variant of Adam based on higher order moments, decoupling weight decay, and decaying momentum).

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        epochs (int): number of epochs the model will be trained on
        setps_per_epoch (int): number of steps taken each epoch. Can be found with len(dataloader)
        lr (float, optional): learning rate (default: 2e-3)
        betas (Tuple[float, float], optional): coefficients used for computing running averages of gradient and a higher order moment
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (default: 0.01)
        k (int, optional): Order of moment. Must be an even integer to conserve convergence. k=2 reverts to the Adam optimizer (default: 2)

        https://arxiv.org/pdf/1910.06878.pdf (HAdam)
        https://arxiv.org/abs/1711.05101 (AdamW)
        https://arxiv.org/pdf/1910.04952.pdf (Demon)
    """

    def __init__(self, params, epochs, steps_per_epoch, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, k=2):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not isinstance(k, int) or not k%2==0 or not 0 < k:
            raise ValueError("Invalid k value: {}".format(k))

        self.T = epochs*steps_per_epoch
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, k=k)
        super(HAdamWDemon, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adamax does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_k'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_k = state['exp_avg'], state['exp_k']
                beta1, beta2 = group['betas']
                eps = group['eps']
                k = group['k']
                lr = group['lr']

                state['step'] += 1
                
                # Weight Decay
                p.mul_(1-lr*group['weight_decay'])
                
                #Momentum Decay (Demon)
                temp_i = 1-(state['step']/self.T)
                beta1 = beta1*temp_i/ ((1 - beta1) + beta1 * temp_i)

                # Update biased first moment estimate.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased k-moment estimate
                exp_k.mul_(beta2).add_(grad**k, alpha=1 - beta2)
                # Update the exponentially weighted infinity norm.
                #norm_buf = torch.cat([
                #    exp_inf.mul_(beta2).unsqueeze(0),
                #    grad.abs().add_(eps).unsqueeze_(0)
                #], 0)
                #torch.amax(norm_buf, 0, keepdim=False, out=exp_inf)

                bias_correction_1 = 1 - beta1 ** state['step']
                bias_correction_2 = (1-beta2**state['step']) ** (1/k)
                
                clr = group['lr'] * bias_correction_2 / bias_correction_1

                p.addcdiv_(exp_avg, exp_k.pow(1/k)+eps, value=-clr)

        return loss

