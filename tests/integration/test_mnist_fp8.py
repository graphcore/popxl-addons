# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import popxl.ops as ops
from typing import Dict, List, Tuple, Mapping
import numpy as np
import torch
from tqdm import tqdm
import popxl
import popxl.ops as ops
from mnist_utils import Timer, get_mnist_data
from popxl_addons.layers.linear_fp8 import LinearFP8
import popxl_addons as addons
import pytest
import os


class LinearFP8GELU(addons.Module):
    def __init__(self, opts):
        """
        Define a FP8 linear layer with GELU activation function in PopXL.
        """
        super().__init__()
        self.opts = opts

    def build(self, x: popxl.Tensor):
        """
        Define the forward pass.
        """
        (y,) = self.call_module(
            LinearFP8(
                out_features=self.opts["n_classes"],
                bias=True,
                scale_metric=self.opts["scale_metric"],
            ),
            name="linear",
        )(x)
        z = ops.gelu(y)
        return z


def update_weights_bias(opts, grads, vars):
    """
    Update weights and bias by w += - lr * grads_w, b += - lr * grads_b.
    """
    for _, var in vars.to_dict().items():
        ops.scaled_add_(var, grads[var], b=-opts["lr"])


def build_train_ir(opts):
    """
    Build the IR for training.
        - load input data
        - buid the network forward pass
        - calculating the gradients, and
        - finally update weights and bias
        - store output data
    """
    ir = popxl.Ir()
    with ir.main_graph, popxl.in_sequence():
        # Host load input and labels
        img_stream = popxl.h2d_stream([opts["batch_size"], 28, 28], popxl.float32, name="input_stream")
        x = ops.host_load(img_stream, "x")
        x = x.reshape((-1, 28 * 28))
        label_stream = popxl.h2d_stream([opts["batch_size"]], popxl.int32, name="label_stream")
        labels = ops.host_load(label_stream, "labels")

        # Build forward pass graph
        linear = LinearFP8GELU(opts)
        facts, linear_graph = linear.create_graph(x)

        # Call autodiff in forward pass graph
        grad_graph = linear_graph.autodiff()
        vars = facts.init()
        fwd_call = linear_graph.bind(vars).call_with_info(x)
        output, *_ = fwd_call.outputs

        # Calculate loss and initial gradients
        probs = ops.softmax(output, axis=-1)
        loss, dy = ops.nll_loss_with_softmax_grad(probs, labels)

        # Build backward pass graph to calculate gradients
        grads_call_info = grad_graph.call_with_info(dy, args=grad_graph.grad_graph_info.inputs_dict(fwd_call))

        # Find the corresponding gradient w.r.t. the input, weights and bias
        grads = grad_graph.grad_graph_info.fwd_parent_ins_to_grad_parent_outs(fwd_call, grads_call_info)

        # Update weights and bias
        update_weights_bias(opts, grads, vars)

        # Host store to get loss
        loss_stream = popxl.d2h_stream(loss.shape, loss.dtype, name="loss_stream")
        ops.host_store(loss_stream, loss)
        output_stream = dict()
        output_stream["loss"] = loss_stream

    return ir, (img_stream, label_stream), vars, output_stream


def build_validation_ir(opts):
    """
    Build the IR for testing.
    """
    ir = popxl.Ir()
    with ir.main_graph, popxl.in_sequence():
        # Host load input and labels
        img_stream = popxl.h2d_stream([opts["test_batch_size"], 28, 28], popxl.float32, name="input_stream")
        x = ops.host_load(img_stream, "x")
        x = x.reshape((-1, 28 * 28))

        # Build forward pass graph
        linear = LinearFP8GELU(opts)
        facts, linear_graph = linear.create_graph(x)
        vars = facts.init()
        fwd_call = linear_graph.bind(vars).call_with_info(x)
        output, *_ = fwd_call.outputs

        # Host store to get loss
        out_stream = popxl.d2h_stream(output.shape, output.dtype, name="loss_stream")
        ops.host_store(out_stream, output)

    return ir, img_stream, out_stream, vars


def train(train_session, training_data, opts, input_streams, output_stream):
    """
    Set up training loop
    """
    nb_batches = len(training_data)
    for epoch in range(1, opts["epochs"] + 1):
        print(f"Epoch {epoch}/{opts['epochs']}")
        bar = tqdm(training_data, total=nb_batches)
        for data, labels in bar:
            inputs: Mapping[popxl.HostToDeviceStream, np.ndarray] = dict(
                zip(
                    input_streams,
                    [data.squeeze().numpy(), labels.int().numpy()],
                )
            )
            outputs = train_session.run(inputs)
            loss = outputs[output_stream["loss"]]
            bar.set_description(f"Average loss: {np.mean(loss):.4f}")


def get_accuracy(predictions: np.ndarray, labels: np.ndarray):
    """
    Calculate the accuracy of predictions.
    """
    ind = np.argmax(predictions, axis=-1).flatten()
    labels = labels.detach().numpy().flatten()
    return np.mean(ind == labels) * 100.0


def compute_validation_accuracy(test_session, test_data, opts, input_streams, out_stream):
    """
    Compute accuracy of validation set
    """
    nr_batches = len(test_data)
    sum_acc = 0.0
    with torch.no_grad():
        for data, labels in tqdm(test_data, total=nr_batches):
            inputs: Mapping[popxl.HostToDeviceStream, np.ndarray] = {input_streams: data.squeeze().numpy()}
            output = test_session.run(inputs)
            sum_acc += get_accuracy(output[out_stream], labels)
    print(f"Accuracy on test set: {sum_acc / len(test_data):0.2f}%")
    return sum_acc / len(test_data)


@pytest.mark.parametrize("scale_metric", ["amax", "manual_scale"])  # mse not included since it takes more than 10 mins
def test_mnist_accuracy(scale_metric):
    opts = {}
    opts["batch_size"] = 32
    opts["test_batch_size"] = 32
    opts["epochs"] = 2
    opts["lr"] = 0.005
    opts["datsets_dir"] = os.environ.get("MNIST_DATASET_LOCATION", ".graphcore/datasets")
    opts["n_classes"] = 10
    opts["scale_metric"] = scale_metric

    np.random.seed(0)  # Fix seed to have deterministic initialization

    # Get the data for training and validation
    training_data, test_data = get_mnist_data(opts["datsets_dir"], opts["batch_size"], opts["test_batch_size"], None)

    with Timer(desc="MNIST training"):
        # Build the ir for training
        train_ir, input_streams, train_variables, output_stream = build_train_ir(opts)
        # session begin
        train_session = popxl.Session(train_ir, "ipu_hw")
        with train_session:
            train(train_session, training_data, opts, input_streams, output_stream)
        # session end
        print("Training complete.")

    with Timer(desc="MNIST testing"):
        # Build the ir for testing
        test_ir, test_input_streams, out_stream, test_variables = build_validation_ir(opts)
        test_session = popxl.Session(test_ir, "ipu_hw")
        # Get test variable values from trained weights
        train_vars_to_data = train_session.get_tensors_data(train_variables.tensors)
        train_vars_to_test_vars = train_variables.to_mapping(test_variables)
        test_vars_to_data = {
            test_var: train_vars_to_data[train_var].copy() for train_var, test_var in train_vars_to_test_vars.items()
        }
        test_session.write_variables_data(test_vars_to_data)
        # test begins
        with test_session:
            acc = compute_validation_accuracy(test_session, test_data, opts, test_input_streams, out_stream)
        # test end
        print("Testing complete.")

        # Check that test accuracy is greater than 0.9
        np.testing.assert_array_less(0.9, acc / 100)
        return acc / 100
