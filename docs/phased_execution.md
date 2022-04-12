# Phased Execution

Phased execution is a method of storing the required inputs (activations and variables) for calling a Graph
in remote memory.

For example, given the Graph:
```python
class Linear(addons.Module):
    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        w = self.add_variable_input("w", partial(np.random.normal, 0, 0.1, (2, 2)), x.dtype)
        y = x @ w
        return y

args, graph = Linear().create_graph(x_spec)
```
when we initialise `w` it creates a variable
```python
variables = args.init()  # NamedTensor(w=popxl.Variable)
```
This variable is typically always-live. Meaning that it will take up space in the memory on chip for the full duration of our program.
When scaling to models with a large number of parameters this will become a bottleneck.

The IPU systems come with associated external memory (popxl.RemoteBuffer) that can be used to store Tensors. This includes the variables in our graph.
```python
buffers = named_variable_buffers(args)
variables = args.init_remote(buffers, 0)
```
then to use the variables we must load them:
```python
load_graph, names = load_remote_graph(buffers)
ts = load_graph.call(0)
vs = NamedTensors.pack(names, ts)
```

## Replica Sharding
To improve the performance of loading from external memory we can shard our variables across replicas of our graph.
This can be controlled by the argument `sharded_threshold` of `named_variable_buffers`, or using `Module.add_replica_sharded_variable_input`.
```python
buffers = named_variable_buffers(args, sharded_threshold=16)
variables = args.init_remote(buffers, 0)
```
To gain access to the full variable it must be all gathered. This is where the performance improvement occurs as the bandwidth between IPUs is greater than access to external memory. 
```python
load_graph, names = load_remote_graph(buffers)

ts = load_graph.call(0)
v_shards = NamedTensors.pack(names, ts)

gather_graph, names = all_gather_replica_sharded_graph(v_shards)

ts = gather_graph.bind(v_shards).call()
vs = NamedTensors.pack(names, ts)
```

## Batch Serialisation
When training a model Gradient Accumulation is used to increase the `global-batch-size` used in the optimizer. Typically this would look like:
```python
# psuedo-code
for _ in range(GA):
    fwd
    loss
    bwd
optimiser
```

When using Phased execution our forward and backward graphs looks like:
```python
# Layer A
vs = load()
xs = load()
ys = a.bind(vs).call(xs)
store(ys)

# Layer B
vs = load()
ys = load()
ys = b.bind(vs).call(xs)
store(ys)
```
This is beneficial to memory as the variables of layer `A` are not live at the same time as layer `B`.
However, if we added a gradient accumulation loop to this we would end up with `GA` times more loading of the variables per optimizer step:
```python
for _ in range(gradient_accumulation_steps):
    ...
    # Layer A
    vs = load()
    xs = load()
    ys = a.bind(vs).call(xs)
    store(ys)

    # Layer B
    vs = load()
    ys = load()
    ys = b.bind(vs).call(xs)
    store(ys)
    ...
optimiser
```
Instead we can rearrange the loop such that we don't change the amount of variable loading.
```python
...
# Layer A
vs = load()
for _ in range(gradient_accumulation_steps):
    xs = load()
    ys = a.bind(vs).call(xs)
    store(ys)

# Layer B
vs = load()
for _ in range(gradient_accumulation_steps):
    vs = load()
    ys = load()
    ys = b.bind(vs).call(xs)
    store(ys)
...
optimiser
```
