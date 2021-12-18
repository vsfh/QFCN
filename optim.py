from torch.optim import Optimizer
import torch


class Quantum_SGD(Optimizer):
    def __init__(self, params, lr=0, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, noise=0.01): # Here we have added the 'noise' parameter (set 0 by default)
        if lr < 0.0: # 'required' was removed here
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if noise < 0.0:
            raise ValueError("Invalid noise value: {}".format(noise))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay,
                        nesterov=nesterov, noise=noise) # Here we have added the 'noise' parameter
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(Quantum_SGD, self).__init__(params, defaults)
    def __setstate__(self, state):
        super(Quantum_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
    def step(self, closure=None):
        """Performs a single optimization step.
            HERE WE ADD THE QUANTUM NOISE!
            implement as relative error (l2 norm)
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            noise = group['noise'] # Here we have added the 'noise' parameter
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                # Classical update rule p = p +(-learning_rate)*p.grad
                # now changed to quatnum
                # WITH RELATIVE ERROR
                # where noise_tensor is a tensor of shape p.shape,
                # noise_tensor has random values distributed as gaussian with std = noise
                noise_tensor = p.shape
                noise_tensor = p.data.new(p.size()).normal_(0, noise).clone().detach().requires_grad_(True)
                #noisy grad entry (i,j) should be grad_ij+noise_ij)*norm_l2(grad)
                norml2_dp = torch.norm(d_p)
                noisy_grad = torch.add(d_p , norml2_dp*noise_tensor)
                #update of parameters
                p.data.add_(d_p, alpha=-group['lr'])
        return loss