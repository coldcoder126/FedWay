import torch
import torch.optim as optim


class omega_update(optim.SGD):

    def __init__(self, params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(omega_update, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)

    def __setstate__(self, state):
        super(omega_update, self).__setstate__(state)

    def step(self, reg_params, batch_index, batch_size, use_gpu, closure=None):
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

                if p in reg_params:
                    grad_data = p.grad.data

                    # The absolute value of the grad_data that is to be added to omega
                    grad_data_copy = p.grad.data.clone()
                    grad_data_copy = grad_data_copy.abs()

                    param_dict = reg_params[p]

                    omega = param_dict['omega']
                    omega = omega.to(torch.device("cuda:0" if use_gpu else "cpu"))

                    current_size = (batch_index + 1) * batch_size
                    step_size = 1 / float(current_size)

                    # Incremental update for the omega
                    omega = omega + step_size * (grad_data_copy - batch_size * (omega))

                    param_dict['omega'] = omega

                    reg_params[p] = param_dict

        return loss


def compute_omega_grads_norm(model, dataloader, optimizer, use_gpu):
    """
    Inputs:
    1) model: A reference to the model for which omega is to be calculated
    2) dataloader: A dataloader to feed the data to the model
    3) optimizer: An instance of the "omega_update" class
    4) use_gpu: Flag is set to True if the model is to be trained on the GPU

    Outputs:
    1) model: An updated reference to the model is returned

    Function: Global version for computing the l2 norm of the function (neural network's) outputs. In
    addition to this, the function also accumulates the values of omega across the items of a task

    """
    # Alexnet object
    model.tmodel.eval()

    index = 0
    for data in dataloader:

        # get the inputs and labels
        inputs, labels = data

        if (use_gpu):
            device = torch.device("cuda:0" if use_gpu else "cpu")
            inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # get the function outputs
        outputs = model.tmodel(inputs)
        del inputs

        # compute the sqaured l2 norm of the function outputs
        l2_norm = torch.norm(outputs, 2, dim=1)
        del outputs

        squared_l2_norm = l2_norm ** 2
        del l2_norm

        sum_norm = torch.sum(squared_l2_norm)
        del squared_l2_norm

        # compute gradients for these parameters
        sum_norm.backward()

        # optimizer.step computes the omega values for the new batches of data
        optimizer.step(model.reg_params, index, labels.size(0), use_gpu)
        del labels

        index = index + 1

    return model


class local_sgd(optim.SGD):
    def __init__(self, params, reg_lambda, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(local_sgd, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.reg_lambda = reg_lambda

    def __setstate__(self, state):
        super(local_sgd, self).__setstate__(state)

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
                    curr_param_value = curr_param_value.cuda()

                    init_val = init_val.cuda()
                    omega = omega.cuda()

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
