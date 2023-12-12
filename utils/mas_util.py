#!/usr/bin/env python
# coding: utf-8


from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import copy
import os
import shutil

import sys
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_reg_params(model, freeze_layers=[]):
    """
    Input:
    1) model: A reference to the model that is being trained
    2) use_gpu: Set the flag to True if the model is to be trained on the GPU
    3) freeze_layers: A list containing the layers for which omega is not calculated. Useful in the
        case of computational limitations where computing the importance parameters for the entire model
        is not feasible

    Output:
    1) model: A dictionary containing importance weights (omega), init_val (keep a reference
    to the initial values of the parameters) for all trainable parameters is calculated and the updated
    model with these reg_params is returned.


    Function: Initializes the reg_params for a model for the initial task (task = 1)

    """

    reg_params = {}

    for name, param in model.named_parameters():
        if not name in freeze_layers:
            # print("Initializing omega values for layer", name)
            omega = torch.zeros(param.size())
            omega = omega.to(device)

            init_val = param.data.clone()
            param_dict = {}

            # for first task, omega is initialized to zero
            param_dict['omega'] = omega
            param_dict['init_val'] = init_val

            # the key for this dictionary is the name of the layer
            reg_params[param] = param_dict

    model.reg_params = reg_params

    return model


def init_reg_params_across_tasks(model, freeze_layers=[]):
    """
    Input:
    1) model: A reference to the model that is being trained
    2) use_gpu: Set the flag to True if the model is to be trained on the GPU
    3) freeze_layers: A list containing the layers for which omega is not calculated. Useful in the
        case of computational limitations where computing the importance parameters for the entire model
        is not feasible

    Output:
    1) model: A dictionary containing importance weights (omega), init_val (keep a reference
    to the initial values of the parameters) for all trainable parameters is calculated and the updated
    model with these reg_params is returned.


    Function: Initializes the reg_params for a model for other tasks in the sequence (task != 1)
    """

    # Get the reg_params for the model

    reg_params = model.reg_params

    for name, param in model.named_parameters():

        if not name in freeze_layers:

            if param in reg_params:
                param_dict = reg_params[param]
                # print("Initializing the omega values for layer for the new task", name)

                # Store the previous values of omega
                prev_omega = param_dict['omega']

                # Initialize a new omega
                new_omega = torch.zeros(param.size())
                new_omega = new_omega.to(device)

                init_val = param.data.clone()
                init_val = init_val.to(device)

                param_dict['prev_omega'] = prev_omega
                param_dict['omega'] = new_omega

                # store the initial values of the parameters
                param_dict['init_val'] = init_val

                # the key for this dictionary is the name of the layer
                reg_params[param] = param_dict

    model.reg_params = reg_params

    return model


def consolidate_reg_params(model):
    """
    Input:
    1) model: A reference to the model that is being trained
    2) use_gpu: Set the flag to True if you wish to train the model on a GPU

    Output:
    1) reg_params: A dictionary containing importance weights (omega), init_val (keep a reference
    to the initial values of the parameters) for all trainable parameters


    Function: This function updates the value (adds the value) of omega across the tasks that the model is
    exposed to

    """
    # Get the reg_params for the model
    reg_params = model.reg_params

    for name, param in model.named_parameters():
        if param in reg_params:
            param_dict = reg_params[param]
            # print("Consolidating the omega values for layer", name)

            # Store the previous values of omega
            prev_omega = param_dict['prev_omega']
            new_omega = param_dict['omega']

            new_omega = torch.add(prev_omega, new_omega)
            del param_dict['prev_omega']

            param_dict['omega'] = new_omega

            # the key for this dictionary is the name of the layer
            reg_params[param] = param_dict

    model.reg_params = reg_params

    return model


def compute_omega_grads_norm(model, dataloader, optimizer):
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
    model.eval()

    index = 0
    for data in dataloader:

        # get the inputs and labels
        inputs, labels = data


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
        optimizer.step(model.reg_params, index, labels.size(0))
        del labels

        index = index + 1

    return model


# need a different function for grads vector
def compute_omega_grads_vector(model, dataloader, optimizer, use_gpu):
    """
    Inputs:
    1) model: A reference to the model for which omega is to be calculated
    2) dataloader: A dataloader to feed the data to the model
    3) optimizer: An instance of the "omega_update" class
    4) use_gpu: Flag is set to True if the model is to be trained on the GPU

    Outputs:
    1) model: An updated reference to the model is returned

    Function: This function backpropagates across the dimensions of the  function (neural network's)
    outputs. In addition to this, the function also accumulates the values of omega across the items
    of a task. Refer to section 4.1 of the paper for more details regarding this idea

    """

    # Alexnet object
    model.train(False)
    model.eval(True)

    index = 0

    for dataloader in dset_loaders:
        for data in dataloader:

            # get the inputs and labels
            inputs, labels = data

            if (use_gpu):
                device = torch.device("cuda:0")
                inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # get the function outputs
            outputs = model.tmodel(inputs)

            for unit_no in range(0, outputs.size(1)):
                ith_node = outputs[:, unit_no]
                targets = torch.sum(ith_node)

                # final node in the layer
                if (node_no == outputs.size(1) - 1):
                    targets.backward()
                else:
                    # This retains the computational graph for further computations
                    targets.backward(retain_graph=True)

                optimizer.step(model.reg_params, False, index, labels.size(0), use_gpu)

                # necessary to compute the correct gradients for each batch of data
                optimizer.zero_grad()

            optimizer.step(model.reg_params, True, index, labels.size(0), use_gpu)
            index = index + 1

    return model


# sanity check for the model to check if the omega values are getting updated
def sanity_model(model):
    for name, param in model.named_parameters():

        print(name)

        if param in model.reg_params:
            param_dict = model.reg_params[param]
            omega = param_dict['omega']

            print("Max omega is", omega.max())
            print("Min omega is", omega.min())
            print("Mean value of omega is", omega.min())


# function to freeze selected layers
def create_freeze_layers(model, no_of_layers=2):
    """
    Inputs
    1) model: A reference to the model
    2) no_of_layers: The number of convolutional layers that you want to freeze in the convolutional base of
        Alexnet model. Default value is 2

    Outputs
    1) model: An updated reference to the model with the requires_grad attribute of the
              parameters of the freeze_layers set to False
    2) freeze_layers: Creates a list of layers that will not be involved in the training process

    Function: This function creates the freeze_layers list which is then passed to the `compute_omega_grads_norm`
    function which then checks the list to see if the omegas need to be calculated for the parameters of these layers

    """

    # The require_grad attribute for the parameters of the classifier layer is set to True by default
    for param in model.classifier.parameters():
        param.requires_grad = True

    for param in model.features.parameters():
        param.requires_grad = False

    # return an empty list if you want to train the entire model
    if (no_of_layers == 0):
        return []

    temp_list = []
    freeze_layers = []

    # get the keys for the conv layers in the model
    for key in model.features._modules:
        if (type(model.features._modules[key]) == torch.nn.modules.conv.Conv2d):
            temp_list.append(key)

    num_of_frozen_layers = len(temp_list) - no_of_layers

    # set the requires_grad attribute to True for the layers you want to be trainable
    for num in range(0, num_of_frozen_layers):
        # pick the layers from the end
        temp_key = temp_list[num]

        for param in model.features[int(temp_key)].parameters():
            param.requires_grad = True

        name_1 = 'features.' + temp_key + '.weight'
        name_2 = 'features.' + temp_key + '.bias'

        freeze_layers.append(name_1)
        freeze_layers.append(name_2)

    return [model, freeze_layers]




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


class omega_vector_update(optim.SGD):

    def __init__(self, params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(omega_vector_update, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)

    def __setstate__(self, state):
        super(omega_vector_update, self).__setstate__(state)

    def step(self, reg_params, finality, batch_index, batch_size, use_gpu, closure=None):
        loss = None

        device = torch.device("cuda:0" if use_gpu else "cpu")

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

                    if not finality:

                        if 'temp_grad' in reg_params.keys():
                            temp_grad = param_dict['temp_grad']

                        else:
                            temp_grad = torch.FloatTensor(p.data.size()).zero_()
                            temp_grad = temp_grad.to(device)

                        temp_grad = temp_grad + grad_data_copy
                        param_dict['temp_grad'] = temp_grad

                        del temp_data


                    else:

                        # temp_grad variable
                        temp_grad = param_dict['temp_grad']
                        temp_grad = temp_grad + grad_data_copy

                        # omega variable
                        omega = param_dict['omega']
                        omega.to(device)

                        current_size = (batch_index + 1) * batch_size
                        step_size = 1 / float(current_size)

                        # Incremental update for the omega
                        omega = omega + step_size * (temp_grad - batch_size * (omega))

                        param_dict['omega'] = omega

                        reg_params[p] = param_dict

                        del omega
                        del param_dict

                    del grad_data
                    del grad_data_copy

        return loss


def exp_lr_scheduler(optimizer, epoch, init_lr=0.0008, lr_decay_epoch=20):
    """
    Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.

    """
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))
    print('lr is ' + str(lr))

    if (epoch % lr_decay_epoch == 0):
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def model_criterion(preds, labels):
    """
    Function: Model criterion to train the model

    """
    loss = nn.CrossEntropyLoss()
    return loss(preds, labels)


def check_checkpoints(store_path):
    """
    Inputs
    1) store_path: The path where the checkpoint file will be searched at

    Outputs
    1) checkpoint_file: The checkpoint file if it exists
    2) flag: The flag will be set to True if the directory exists at the path

    Function: This function takes in the store_path and checks if a prior directory exists
    for the task already. If it doesn't, flag is set to False and the function returns an empty string.
    If a directory exists the function returns a checkpoint file

    """

    # if the directory does not exist return an empty string
    if not os.path.isdir(store_path):
        return ["", False]

    # directory exists but there is no checkpoint file
    onlyfiles = [f for f in os.listdir(store_path) if os.path.isfile(os.path.join(store_path, f))]
    max_train = -1
    flag = False

    # Check the latest epoch file that was created
    for file in onlyfiles:
        if (file.endswith('pth.tr')):
            flag = True
            test_epoch = file[0]
            if (test_epoch > max_train):
                max_epoch = test_epoch
                checkpoint_file = file

    # no checkpoint exists in the directory so return an empty string
    if (flag == False):
        checkpoint_file = ""

    return [checkpoint_file, True]


def create_task_dir(task_no, no_of_classes, store_path):
    """
    Inputs
    1) task_no: The identity for the task defined by it's number in the sequence
    2) no_of_classes: The number of classes that the particular task has


    Function: This function creates a directory to store the classification head for the new task. It also
    creates a text file which stores the number of classes that this task contained.

    """

    os.mkdir(store_path)
    file_path = os.path.join(store_path, "classes.txt")

    with open(file_path, 'w') as file1:
        input_to_txtfile = str(no_of_classes)
        file1.write(input_to_txtfile)
        file1.close()

    return


def model_inference(task_no, use_gpu=False):
    """
    Inputs
    1) task_no: The task number for which the model is being evaluated
    2) use_gpu: Set the flag to True if you want to run the code on GPU. Default value: False

    Outputs
    1) model: A reference to the model

    Function: Combines the classification head for a particular task with the shared model and
    returns a reference to the model is used for testing the process

    """

    # all models are derived from the Alexnet architecture
    pre_model = models.alexnet(pretrained=True)
    model = shared_model(pre_model)

    path_to_model = os.path.join(os.getcwd(), "models")

    path_to_head = os.path.join(os.getcwd(), "models", "Task_" + str(task_no))

    # get the number of classes by reading from the text file created during initialization for this task
    file_name = os.path.join(path_to_head, "classes.txt")
    file_object = open(file_name, 'r')
    num_classes = file_object.read()
    file_object.close()

    num_classes = int(num_classes)
    # print (num_classes)
    in_features = model.classifier[-1].in_features

    del model.classifier[-1]
    # load the classifier head for the given task identified by the task number
    classifier = classification_head(in_features, num_classes)
    classifier.load_state_dict(torch.load(os.path.join(path_to_head, "head.pth")))

    # load the trained shared model
    model.load_state_dict(torch.load(os.path.join(path_to_model, "shared_model.pth")))

    model.classifier.add_module('6', nn.Linear(in_features, num_classes))

    # change the weights layers to the classifier head weights
    model.classifier[-1].weight.data = classifier.fc.weight.data
    model.classifier[-1].bias.data = classifier.fc.bias.data

    # device = torch.device("cuda:0" if use_gpu else "cpu")
    model.eval()
    # model.to(device)

    return model


def model_init(no_classes, use_gpu=False):
    """
    Inputs
    1) no_classes: The number of classes that the model is exposed to in the new task
    2) use_gpu: Set the flag to True if you want to run the code on GPU. Default value = False

    Outputs
    1) model: A reference to the model that has been initialized

    Function: Initializes a model for the new task which the shared features and a classification head
    particular to the new task

    """

    path = os.path.join(os.getcwd(), "models", "shared_model.pth")
    path_to_reg = os.path.join(os.getcwd(), "models", "reg_params.pickle")

    pre_model = models.alexnet(pretrained=True)
    model = shared_model(pre_model)

    # initialize a new classification head
    in_features = model.classifier[-1].in_features

    del model.classifier[-1]

    # load the model
    if os.path.isfile(path):
        model.load_state_dict(torch.load(path))

    # add the last classfication head to the shared model
    model.classifier.add_module('6', nn.Linear(in_features, no_classes))

    # load the reg_params stored
    if os.path.isfile(path_to_reg):
        with open(path_to_reg, 'rb') as handle:
            reg_params = pickle.load(handle)

        model.reg_params = reg_params

    device = torch.device("cuda:0" if use_gpu else "cpu")
    model.train(True)
    model.to(device)

    return model

