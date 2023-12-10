import torch
import torch.optim as optim


class optimizer_mas(optim.SGD):
    def __init__(self, params, reg_lambda, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(optimizer_mas, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.reg_lambda = reg_lambda

    def __setstate__(self, state):
        super(optimizer_mas, self).__setstate__(state)

    def step(self, reg_params, closure=None):

        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:

                if p.grad is None:
                    continue

                d_p = p.grad.data

                if p in reg_params:
                    param_dict = reg_params[p]

                    omega = param_dict['omega']
                    init_val = param_dict['init_val']

                    curr_param_value = p.data
                    curr_param_value = curr_param_value

                    init_val = init_val
                    omega = omega

                    # get the difference
                    param_diff = curr_param_value - init_val

                    # get the gradient for the penalty term for change in the weights of the parameters
                    local_grad = torch.mul(param_diff, 2 * self.reg_lambda * omega)

                    del param_diff
                    del omega
                    del init_val
                    del curr_param_value

                    d_p = d_p + local_grad

                    del local_grad

                if (weight_decay != 0):
                    d_p.add_(weight_decay, p.data)

                if (momentum != 0):
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss
