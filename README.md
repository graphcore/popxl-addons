# popart-ir-extensions

`popart.ir` extensions.

Obligatory package alias:

```python
import popart_ir_extensions as pir_ext
```

For examples please see the user guide or tests.

## Development

* To reformat code to repo standard: `make lint`
* To build documentation: `make docs`
* Do not push to master branch. Make changes through github PR requests.

## User guide

### Graph concepts

The `Graphs` module includes extension to help with managing pir graphs, inputs and defining variables.

There are three types of graphs GenericGraph, ConcreteGraph and CallableGraph.
You define a GenericGraph and this then evolves to a CallableGraph as so:
```
┌──────────────┐ Outlined  ┌───────────────┐          ┌───────────────┐
│              ├──────────►│               │          │               │
│ GenericGraph │           │ ConcreteGraph ├─────────►│ CallableGraph │
│              ├──────────►│               │          │               │
└──────────────┘ Inlined   └───────────────┘          └───────────────┘
```

* **GenericGraph**:
  * Conceptually a graph where the tensor shapes and datatypes are not known
  * Includes a `build` method which creates a `ConcreteGraph` usually using an input tensor to provide the shape and datatype
  * The build method can also includes input definitions which provide a factory method for creating input variables
  * A GenericGraph can include sub-GenericGraphs or sub-ConcreteGraphs that can either be inlined or outlined - see below examples
  * To build a GenericGraph from the GenericGraph the build method invokes the method `ir.create_graph`
* **ConcreteGraph**:
  * Conceptually a graph with defined tensor shapes and datatypes
  * A subclass of `pir.Graph` which also includes input definitions inherited from the parent GenericGraph
  * To build a CallableGraph from the GenericGraph the user has the option to create variables using the input factories
* **CallableGraph**:
  * Conceptually a graph with defined tensor shapes and datatypes, and defined inputs
  * A object which has a `pir.Graph` and a map between graph inputs and tensors
  * Can be called using the `call` method

#### `pir` vs `pir_ext` examples

**Example 1**
* A subgraph is created that performs scale
* The subgraph is called twice to before the operation twice with the same input varibles
* The code is outlined as you are reusing the same subgraph

`pir`:
```python
import numpy as np
import popart.ir as pir
from popart.ir import ops
import popart_ir_extensions as pir_ext

def scale_fn(x: pir.Tensor, scale: pir.Tensor):
    return x * scale

ir = pir.Ir()
main = ir.main_graph()
with main:
    x_h2d = pir.h2d_stream((2, 2), pir.float32, name="x_stream")
    x = ops.host_load(x_h2d, "x")

    scale = pir.variable(np.ones(x.shape, x.dtype.as_numpy()), name="scale")
    scale_graph = ir.create_graph(scale_fn, x, scale)

    y = ops.call(scale_graph, x, scale) # Subgraph A. Add subgraph to maingraph. Call site 1
    z = ops.call(scale_graph, y, scale) # Subgraph A. Call site 2
```

`pir_ext`:
```python
class Scale(pir_ext.GenericGraph):
    def build(self, x: pir.Tensor) -> pir.Tensor:
        self.scale = self.add_input_tensor("scale", lambda: np.ones(x.shape, x.dtype.as_numpy()))
        return x * self.scale

ir = pir.Ir()
main = ir.main_graph()
with main:
        x_h2d = pir.h2d_stream((2, 2), pir.float32, name="x_stream")
        x = ops.host_load(x_h2d, "x")

        scale_graph = Scale().to_concrete(x)
        scale = scale_graph.to_callable(create_inputs=True)

        y = scale.call(x) # Subgraph A
        z = scale.call(y) # Subgraph A
```

**Example 2**:
* Reuse of the same subgraph but with a different scale variable

`pir`:
```python
def scale_fn(x: pir.Tensor, scale: pir.Tensor):
        return x * scale

ir = pir.Ir()
main = ir.main_graph()
with main:
        x_h2d = pir.h2d_stream((2, 2), pir.float32, name="x_stream")
        x = ops.host_load(x_h2d, "x")

        scale1 = pir.variable(np.ones(x.shape, x.dtype.as_numpy()), name="scale")
        scale2 = pir.variable(np.ones(x.shape, x.dtype.as_numpy()), name="scale")
        scale_graph = ir.create_graph(scale_fn, x, scale1) #this is only taking shape and type

        y = ops.call(scale_graph, x, scale1) # Subgraph A with scale 1
        z = ops.call(scale_graph, y, scale2) # Subgraph A with scale 2
```

`pir_ext`:
```python
class Scale(pir_ext.GenericGraph):
    def build(self, x: pir.Tensor) -> pir.Tensor:
        self.scale = self.add_input_tensor("scale", lambda: np.ones(x.shape, x.dtype.as_numpy()))
        return x * self.scale

ir = pir.Ir()
main = ir.main_graph()
with main:
        x_h2d = pir.h2d_stream((2, 2), pir.float32, name="x_stream")
        x = ops.host_load(x_h2d, "x")

        scale_graph = Scale().to_concrete(x)
        scale1 = scale_graph.to_callable(create_inputs=True)
        scale2 = scale_graph.to_callable(create_inputs=True)

        y = scale1.call(x) # Subgraph A with scale 1
        z = scale2.call(y) # Subgraph A with scale 2
```

#### Other examples

**Inlining vs outlining sub-GenericGraphs**:

Here is an example of subgraph inlining:
`Scale` is used twice in `InlinedScale`. The GenericGraph detects that the attributes `scale1` and `scale2` are
GenericGraphs and captures the variable definitions. The result is that a single `pir.Graph` is created.
```python
class Scale(pir_ext.GenericGraph):
    def build(self, x: pir.Tensor) -> pir.Tensor:
        self.scale = self.add_input_tensor("scale", lambda: np.ones(x.shape, x.dtype.as_numpy()))
        return x * self.scale

class InlinedScale(pir_ext.GenericGraph):
    def __init__(self):
        super().__init__()
        self.scale1 = Scale()
        self.scale2 = Scale()
    
    def build(self, x: pir.Tensor) -> pir.Tensor:
        x = self.scale1(x)
        x = self.scale2(x)
        return x
```

Here is an example of subgraph outlining: during the build step of the GenericGraph `Scale` is outlined by creating a
ConcreteGraph, which is then used twice to create two callables with different varibles.
```python
class OutlinedScale(pir_ext.GenericGraph):
    def build(self, x: pir.Tensor) -> pir.Tensor:
        self.scale = Scale().to_concrete(x)  # Outline `Scale`
        self.scale1 = self.scale.to_callable(create_inputs=False)
        self.scale2 = self.scale.to_callable(create_inputs=False)
        x = self.scale1.call(x)
        x = self.scale2.call(x)
        return x
```
