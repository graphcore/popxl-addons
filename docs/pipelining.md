# Pipelining

A method of executing partitions of a model in parallel. Consider the loop:
```python
for i in range(10):
  x = ops.host_load(h2d_stream)
  x = layer0(x)
  x = layer1(x)
  ops.host_store(d2h_stream, x)
```
There is a data dependency between `layer0` -> `layer1`. However there is no data dependency between iterations of the loop.
Executing subsets of the total loop iterations on different IPUs is known as data parallel (replication). However, this requires
the variables (and other tensors) of all layers to fit within each IPU's memory. Instead we can decompose the loop and parallelise across layers:
```python
x0 = ops.host_load(h2d_stream)
x0_ = layer0(x0)
x1 = x0_.copy_to_ipu(1)

for i in range(9):
    x0 = ops.host_load(h2d_stream)
    x0_ = layer0(x0)
    x1_ = layer1(x1)
    ops.host_store(d2h_stream, x1_)
    x1 = x0_.copy_to_ipu(1)

x1_ = layer1(x1)
ops.host_store(d2h_stream, x1_)
```
This program has no data dependencies between layer0 and layer1 within the loop. As such we can execute them on different IPUs in parallel.

## Defining a pipeline

Using the `addons` pipelining transformation we can describe how to decompose our loop and execute in parallel.

First we must create a pipelining context:
```python
with addons.pipelined_execution(steps=10) as p:
```
This is similar to `for i in range(10)`. Within this context we will define one step of the loop using `popxl` annotations.
When the context closes, the transformation will be run and the current graph will end up with a single `ops.call` that executes the pipeline.
```python
  with p.stage(0), popxl.ipu(0):
    x = ops.host_load(h2d_stream)
    x = layer0(x)
    x = x.copy_to_ipu(0)

  with p.stage(1), popxl.ipu(1):
    x = layer1(x)
    ops.host_store(d2h_stream, x)
```
Here we have added the `stage` and the `ipu` annotations.
There are no constraints on how many or which ipus a stage can run on. However, any communication between ipus will cause
a syncronisation that can stall the pipeline. In general, we want the data dependencies between stages to only be represented as
`ops.ipu_copy` or `Tensor.copy_to_ipu`.

### Training

When training a model we want to execute the forward and gradient layers on the same IPU. This preference comes from that fact both layers require the parameters of the model, so if they executed on different IPUs we would incure an communication cost to move the parameters around.

To achieve this behaviour we can just reuse the same `ipu` annotation when calling the gradient layer.
```python
  with p.stage(0), popxl.ipu(0):
    x = ops.host_load(h2d_stream)
    x, = layer0.call(x)

...

  with p.stage(2), popxl.ipu(0):
    dlayer0.call(...)
```
Our next concern is how to provide inputs to the gradient layer. Gradient layers are constructed using `addons.autodiff`. This extension returns a convience class for the autodiff result, `ConcreteGradGraph`. Typically a gradient graph has two types of expected inputs:
* `FwdGrad`: The gradients of some outputs of the forward graph.
* `Fwd`: Inputs/Outputs of the forward graph. Often referred to as "activations" (although this may also contain parameters).

Without pipelining we can use the method `addons.connect_activations` to attach expected `Fwd` connections from a callsite of the forward's graph to a `CallableGradGraph` (created from `ConcreteGradGraph.to_callable`).

When pipelining there will be a _delay_ between the execution of the forward and gradient stages. In this delay additional forward execution will happen that will overwrite the activations of the a previous step. To avoid this we must keep activations in a "stash" to be able to "restore" them later.
`Stash` and `Restore` classes have been provided to add the required stash and restore operations. The method `stash_and_restore_activations` calculates the required stash size then add `Stash` graphs to the required forward stage and `Restore` graphs to the gradient stage. This can be used as follows:
```python
with addons.pipelined_execution(10) as p:
  with p.stage(0), popxl.ipu(0):
    x = ops.host_load(h2d_stream)
    call_info = layer0.call_with_info(x)
    x, = call_info.outputs

...

  with p.stage(2), popxl.ipu(0):
    acts = p.stash_and_restore_activations(call_info, grad_info)
    dlayer0.call(dx, args=acts)
```

If you would like more control over when the `Stash` graph is executed, you can use `stash_and_restore_tensor` instead.


