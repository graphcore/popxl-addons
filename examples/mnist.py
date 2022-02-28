# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import argparse
from functools import partial
from typing import Mapping, Optional
import torch
import torchvision
from tqdm import tqdm
import numpy as np
import popart.ir as pir
import popart.ir.ops as ops
import popart_ir_extensions as pir_ext


def get_mnist_data(opts):
    training_data = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        '~/.torch/datasets',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.1307, ), (0.3081, ))])),
                                                batch_size=opts.batch_size,
                                                shuffle=True,
                                                drop_last=True)

    validation_data = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        '~/.torch/datasets',
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.1307, ), (0.3081, ))])),
                                                  batch_size=opts.test_batch_size,
                                                  shuffle=True,
                                                  drop_last=True)
    return training_data, validation_data


class Linear(pir_ext.Module):
    def __init__(self, out_features: int, bias: bool = True):
        super().__init__()
        self.out_features = out_features
        self.bias = bias

    def build(self, x: pir.Tensor) -> pir.Tensor:
        w = self.add_input_tensor("weight", partial(np.random.normal, 0, 0.02, (x.shape[-1], self.out_features)),
                                  x.dtype)
        y = x @ w
        if self.bias:
            b = self.add_input_tensor("bias", partial(np.zeros, y.shape[-1]), x.dtype)
            y = y + b
        return y


class Net(pir_ext.Module):
    def __init__(self, cache: Optional[pir_ext.GraphCache] = None):
        super().__init__(cache=cache)
        self.fc1 = Linear(512)
        self.fc2 = Linear(512)
        self.fc3 = Linear(10)

    def build(self, x: pir.Tensor):
        x = x.reshape((-1, 28 * 28))
        x = ops.gelu(self.fc1(x))
        x = ops.gelu(self.fc2(x))
        x = self.fc3(x)
        return x


def accuracy(predictions: np.ndarray, labels: np.ndarray):
    ind = np.argmax(predictions, axis=-1).flatten()
    labels = labels.detach().numpy().flatten()
    return np.mean(ind == labels) * 100.0


def train(train_runner, training_data, opts, streams):
    nr_batches = len(training_data)
    for epoch in range(1, opts.epochs + 1):
        print("Epoch {0}/{1}".format(epoch, opts.epochs))
        bar = tqdm(training_data, total=nr_batches)
        for data, labels in bar:
            inputs = dict(zip(streams, [data.float(), labels.int()]))
            loss = train_runner.run(inputs)
            bar.set_description("Loss:{:0.4f}".format(loss))


def test(test_runner, test_data, streams):
    nr_batches = len(test_data)
    sum_acc = 0.0
    with torch.no_grad():
        for data, labels in tqdm(test_data, total=nr_batches):
            inputs = dict(zip(streams, [data]))
            output = test_runner.run(inputs)
            sum_acc += accuracy(output, labels)
    print("Accuracy on test set: {:0.2f}%".format(sum_acc / len(test_data)))


def train_program(opts):
    ir = pir.Ir()
    with ir.main_graph:
        # Inputs
        in_stream = pir.h2d_stream((opts.batch_size, 28, 28), pir.float32, "image")
        in_t = ops.host_load(in_stream)
        label_stream = pir.h2d_stream((opts.batch_size, ), pir.int32, "labels")
        labels = ops.host_load(label_stream, "labels")

        # Create graphs
        args, graph = Net().create_graph(in_t)
        dgraph = pir_ext.autodiff(graph)

        # Initialise variables
        params = args.init()

        # Forward
        fwd_info = graph.bind(params).call_with_info(in_t)
        x = fwd_info.outputs[0]
        # Loss
        loss, dx = pir_ext.ops.cross_entropy_with_grad(x, labels)
        loss_stream = pir.d2h_stream(loss.shape, loss.dtype, "loss")
        ops.host_store(loss_stream, loss)
        # Gradient
        bwd_info = dgraph.call_with_info(dx, args=dgraph.grad_graph_info.inputs_dict(fwd_info))
        # Optimizer
        grads = dgraph.grad_graph_info.fwd_parent_ins_to_grad_parent_outs(fwd_info, bwd_info)
        for t in params.tensors:
            ops.scaled_add_(t, grads[t], b=-opts.lr)

    return pir_ext.Runner(ir, loss_stream), [in_stream, label_stream], params


def test_program(opts):
    ir = pir.Ir()
    with ir.main_graph:
        # Inputs
        in_stream = pir.h2d_stream((opts.test_batch_size, 28, 28), pir.float32, "image")
        in_t = ops.host_load(in_stream)

        # Create graphs
        args, graph = Net().create_graph(in_t)

        # Initialise variables
        params = args.init()

        # Forward
        x, = graph.bind(params).call(in_t)
        x_stream = pir.d2h_stream(x.shape, x.dtype, "x")
        ops.host_store(x_stream, x)

    return pir_ext.Runner(ir, x_stream), [in_stream], params


def copy_checkpoint(src: pir_ext.NamedTensors, dst: pir_ext.NamedTensors,
                    ckpt: Mapping[pir.Tensor, pir_ext.HostTensor]) -> Mapping[pir.Tensor, pir_ext.HostTensor]:
    src_dst = src.to_mapping(dst)
    return {src_dst[src_t]: t for src_t, t in ckpt.items()}


def main():
    parser = argparse.ArgumentParser(description='MNIST training in popart.ir extensions')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for training (default: 8)')
    parser.add_argument('--test-batch-size', type=int, default=80, help='batch size for testing (default: 80)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate (default: 0.05)')
    opts = parser.parse_args()

    training_data, test_data = get_mnist_data(opts)

    train_runner, train_inputs, train_params = train_program(opts)

    train(train_runner, training_data, opts, train_inputs)

    trained_params = train_runner.read_weights(train_params.tensors)
    train_runner.detach()

    test_runner, test_inputs, test_params = test_program(opts)

    test_runner.write_weights(copy_checkpoint(train_params, test_params, trained_params))

    test(test_runner, test_data, test_inputs)


if __name__ == '__main__':
    main()
