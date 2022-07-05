# popxl.addons

> **Warning**
> The popxl Python package is currently experimental and may be subject to change in future releases in ways that are backwards incompatible without deprecation warnings.

> **Warning**
> Due to the experimental nature of PopXL and popxl.addons the user and API reference documentation provided is incomplete.

`popxl.addons` provides additionally functionality to supplement the PopXL framework.

Obligatory package alias:

```python
import popxl_addons as addons
```

For examples please see the [tests](tests).

Please note we are not currently accepting pull requests or issues on this repository. If you are actively using this repository and want to report any issues, please raise a ticket through the Graphcore support portal: https://www.graphcore.ai/support.

## Installation

- Install the Poplar SDK >=2.6 and source the enable.sh scripts for both PopART and Poplar as described in the [Getting Started guide](https://docs.graphcore.ai/en/latest/getting-started.html) for your IPU system
- Create a Python virtual environment: `python3 -m venv <virtual_env>`
- Activate the virtual environment: `. <virtual_env>/bin/activate`
- Update `pip`: `pip3 install --upgrade pip`
- If you cloned the `popxl.addons` repo, install its requirements in `popxl-addons` with `pip3 install -r requirements.txt` and add the repo to the python path. Otherwise, you can pip install `popxl.addons`: `pip3 install git+ssh://git@github.com:graphcore/popxl-addons.git`.

## User guide

### Module concepts

The Module extends popxl's create_graph to help with managing variables.

First we define a Module class. 
```
class Scale(addons.Module):
    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        scale = self.add_variable_input("scale", partial(np.ones, x.shape), x.dtype)
        return x * scale
```
Then we create a graph from the module:
```
args, graph = Scale().create_graph(x)
```
A tuple is returned from `Module.create_graph`.

The first value is a `NamedVariableFactories` object. This contains all inputs to the graph that are created using `Module.add_variable_input`
during the construction of the graph. In most cases these can be considered the constructors of variables of your modules. 
If we want an instance of these variables we can initialise one:
```
scale_vars = args.init()
```

The second value is a `GraphWithNamedArgs`. This is a combination of a `popxl.Graph` and `NamedTensors`. The named tensors keep a record
of each input to the graph created using `Module.add_variable_input` and has the same naming as the `NamedVariableFactories` above.
To be able to call this graph we must first provide tensors for each of the named args. This can be done by using `bind`:
```
layer = graph.bind(scale_vars)
```
`layer` is a `BoundGraph`. Which is a combination of a compute graph and some connected inputs. Finally to call the graph we provide any positional arguments:
```
y, = layer.call(x)
```

#### `popxl` vs `addons` examples

**Example 1**
* A subgraph is created that performs scale
* The subgraph is called twice to before the operation twice with the same input varibles
* The code is outlined as you are reusing the same subgraph

`popxl`:

```python
import numpy as np
import popxl
from popxl import ops
import popxl_addons as addons

def scale_fn(x: popxl.Tensor, scale: popxl.Tensor):
    return x * scale

ir = popxl.Ir()
main = ir.main_graph
with main:
    x_h2d = popxl.h2d_stream((2, 2), popxl.float32, name="x_stream")
    x = ops.host_load(x_h2d, "x")

    scale = popxl.variable(np.ones(x.shape, x.dtype.as_numpy()), name="scale")
    scale_graph = ir.create_graph(scale_fn, x, scale)

    y, = ops.call(scale_graph, x, scale) # Subgraph A. Add subgraph to maingraph. Call site 1
    z, = ops.call(scale_graph, y, scale) # Subgraph A. Call site 2
```

`addons`:
```python
class Scale(addons.Module):
    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        self.scale = self.add_variable_input("scale", lambda: np.ones(x.shape, x.dtype.as_numpy()))
        return x * self.scale

ir = popxl.Ir()
main = ir.main_graph
with main:
    x_h2d = popxl.h2d_stream((2, 2), popxl.float32, name="x_stream")
    x = ops.host_load(x_h2d, "x")

    args, graph = Scale().create_graph(x)
    scale = graph.bind(args.init())

    y, = scale.call(x) # Subgraph A
    z, = scale.call(y) # Subgraph A
```

**Example 2**:
* Reuse of the same subgraph but with a different scale variable

`popxl`:
```python
def scale_fn(x: popxl.Tensor, scale: popxl.Tensor):
        return x * scale

ir = popxl.Ir()
main = ir.main_graph
with main:
    x_h2d = popxl.h2d_stream((2, 2), popxl.float32, name="x_stream")
    x = ops.host_load(x_h2d, "x")

    scale1 = popxl.variable(np.ones(x.shape, x.dtype.as_numpy()), name="scale")
    scale2 = popxl.variable(np.ones(x.shape, x.dtype.as_numpy()), name="scale")
    scale_graph = ir.create_graph(scale_fn, x, scale1) #this is only taking shape and type

    y, = ops.call(scale_graph, x, scale1) # Subgraph A with scale 1
    z, = ops.call(scale_graph, y, scale2) # Subgraph A with scale 2
```

`addons`:
```python
class Scale(addons.Module):
    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        self.scale = self.add_variable_input("scale", lambda: np.ones(x.shape, x.dtype.as_numpy()))
        return x * self.scale

ir = popxl.Ir()
main = ir.main_graph
with main:
    x_h2d = popxl.h2d_stream((2, 2), popxl.float32, name="x_stream")
    x = ops.host_load(x_h2d, "x")

    args, graph = Scale().create_graph(x)
    scale1 = graph.bind(args.init())
    scale2 = graph.bind(args.init())

    y, = scale1.call(x) # Subgraph A with scale 1
    z, = scale2.call(y) # Subgraph A with scale 2
```
