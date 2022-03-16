# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import argparse
from functools import partial
from typing import Mapping, Optional
import torch
import torchvision
from tqdm import tqdm
import numpy as np
import popxl
import popxl_addons as addons
import popxl.ops as ops


def get_mnist_data(test_batch_size: int, batch_size: int):
    training_data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            '~/.torch/datasets',
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307, ), (0.3081, )),  # mean and std computed on the training set.
            ])),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

    validation_data = torch.utils.data.DataLoader(torchvision.datasets.MNIST('~/.torch/datasets',
                                                                             train=False,
                                                                             download=True,
                                                                             transform=torchvision.transforms.Compose([
                                                                                 torchvision.transforms.ToTensor(),
                                                                                 torchvision.transforms.Normalize(
                                                                                     (0.1307, ), (0.3081, )),
                                                                             ])),
                                                  batch_size=test_batch_size,
                                                  shuffle=True,
                                                  drop_last=True)
    return training_data, validation_data


def accuracy(predictions: np.ndarray, labels: np.ndarray):
    ind = np.argmax(predictions, axis=-1).flatten()
    labels = labels.detach().numpy().flatten()
    return np.mean(ind == labels) * 100.0


class Linear(addons.Module):
    def __init__(self, out_features: int, bias: bool = True):
        super().__init__()
        self.out_features = out_features
        self.bias = bias

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        # add a state variable to the module
        w = self.add_input_tensor("weight", partial(np.random.normal, 0, 0.02, (x.shape[-1], self.out_features)),
                                  x.dtype)
        y = x @ w
        if self.bias:
            # add a state variable to the module
            b = self.add_input_tensor("bias", partial(np.zeros, y.shape[-1]), x.dtype)
            y = y + b
        return y


class Net(addons.Module):
    def __init__(self, cache: Optional[addons.GraphCache] = None):
        super().__init__(cache=cache)
        self.fc1 = Linear(512)
        self.fc2 = Linear(512)
        self.fc3 = Linear(512)
        self.fc4 = Linear(10)

    def build(self, x: popxl.Tensor):
        x = x.reshape((-1, 28 * 28))
        x = ops.gelu(self.fc1(x))
        x = ops.gelu(self.fc2(x))
        x = ops.gelu(self.fc3(x))
        x = self.fc4(x)
        return x


def train(train_session, training_data, opts, input_streams, loss_stream):
    nr_batches = len(training_data)
    for epoch in range(1, opts.epochs + 1):
        print("Epoch {0}/{1}".format(epoch, opts.epochs))
        bar = tqdm(training_data, total=nr_batches)
        for data, labels in bar:
            inputs: Mapping[popxl.HostToDeviceStream, np.ndarray] = dict(
                zip(input_streams, [addons.utils.to_numpy(data.squeeze().float()),
                                    addons.utils.to_numpy(labels.int())]))
            loss = train_session.run(inputs)
            bar.set_description("Loss:{:0.4f}".format(loss[loss_stream]))


def test(test_session, test_data, input_streams, out_stream):
    nr_batches = len(test_data)
    sum_acc = 0.0
    with torch.no_grad():
        for data, labels in tqdm(test_data, total=nr_batches):
            inputs: Mapping[popxl.HostToDeviceStream, np.ndarray] = dict(
                zip(input_streams, [addons.utils.to_numpy(data.squeeze().float()),
                                    addons.utils.to_numpy(labels.int())]))
            output = test_session.run(inputs)
            sum_acc += accuracy(output[out_stream], labels)
    print("Accuracy on test set: {:0.2f}%".format(sum_acc / len(test_data)))


def train_program(opts):
    ir = popxl.Ir()
    ir.replication_factor = 1

    with ir.main_graph:
        # Create input streams from host to device
        img_stream = popxl.h2d_stream((opts.batch_size, 28, 28), popxl.float32, "image")
        img_t = ops.host_load(img_stream)  #load data
        label_stream = popxl.h2d_stream((opts.batch_size, ), popxl.int32, "labels")
        labels = ops.host_load(label_stream, "labels")

        # Create forward graph
        facts, fwd_graph = Net().create_graph(img_t)
        # Create backward graph via autodiff transform
        bwd_graph = addons.autodiff(fwd_graph)

        # Initialise variables (weights)
        variables = facts.init()

        # Call the forward with call_with_info because we want to retrieve information from the call site
        fwd_info = fwd_graph.bind(variables).call_with_info(img_t)
        x = fwd_info.outputs[0]  # forward output

        # Compute loss and starting gradient for backprop
        loss, dx = addons.ops.cross_entropy_with_grad(x, labels)

        # Setup a stream to retrieve loss values from the host
        loss_stream = popxl.d2h_stream(loss.shape, loss.dtype, "loss")
        ops.host_store(loss_stream, loss)

        # retrieve activations from the forward
        activations = bwd_graph.grad_graph_info.inputs_dict(fwd_info)
        # call the backward providing the starting value for backprop and activations
        bwd_info = bwd_graph.call_with_info(dx, args=activations)

        # Optimizer
        # get a dictionary between forward tensors and corresponding gradients and use it to update
        # each tensor
        grads_dict = bwd_graph.grad_graph_info.fwd_parent_ins_to_grad_parent_outs(fwd_info, bwd_info)
        for t in variables.variables:
            ops.scaled_add_(t, grads_dict[t], b=-opts.lr)

    ir.num_host_transfers = 1
    return popxl.Session(ir, 'ipu_hw'), [img_stream, label_stream], variables, loss_stream


def test_program(opts):
    ir = popxl.Ir()
    ir.replication_factor = 1
    with ir.main_graph:
        # Inputs
        in_stream = popxl.h2d_stream((opts.test_batch_size, 28, 28), popxl.float32, "image")
        in_t = ops.host_load(in_stream)

        # Create graphs
        facts, graph = Net().create_graph(in_t)

        # Initialise variables
        variables = facts.init()

        # Forward
        outputs, = graph.bind(variables).call(in_t)
        out_stream = popxl.d2h_stream(outputs.shape, outputs.dtype, "outputs")
        ops.host_store(out_stream, outputs)

    ir.num_host_transfers = 1
    return popxl.Session(ir, 'ipu_hw'), [in_stream], variables, out_stream


def main():
    parser = argparse.ArgumentParser(description='MNIST training in popxl.addons')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for training (default: 8)')
    parser.add_argument('--test-batch-size', type=int, default=80, help='batch size for testing (default: 80)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate (default: 0.05)')
    opts = parser.parse_args()

    training_data, test_data = get_mnist_data(opts.test_batch_size, opts.batch_size)

    train_session, train_input_streams, train_variables, loss_stream = train_program(opts)

    train(train_session, training_data, opts, train_input_streams, loss_stream)

    trained_weights_data_dict = train_session.get_tensors_data(train_variables.variables)
    train_session.device.detach()

    test_session, test_input_streams, test_variables, out_stream = test_program(opts)
    # Copy trained weights to the program, with a single host to device transfer at the end
    test_session.write_variables_data(dict(zip(test_variables.variables, trained_weights_data_dict.values())))

    #check that weights have been copied correctly
    weights_data_dict = test_session.get_tensors_data(test_variables.variables)
    names_to_data_dic_train = dict(zip(train_variables.named_variables.keys(), trained_weights_data_dict.values()))
    names_to_data_dic_test = dict(zip(test_variables.named_variables.keys(), weights_data_dict.values()))
    for name, array in names_to_data_dic_test.items():
        assert (array == names_to_data_dic_train[name]).all()

    test(test_session, test_data, test_input_streams, out_stream)
    test_session.device.detach()


if __name__ == '__main__':
    main()
