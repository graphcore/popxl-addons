# PopART Extensions
Utilities built on top of PopART 

## Graphs

Extension to help with managing graph inputs and defining variables.

GenericGraph is the class that constructs your graph. This can include constructing variable defintions to be later turned into variables

ConcreteGraph is a implementation of the graph with known shapes and types. This will include variable definitions. 
It is constructed by calling `.to_concrete()` on a GenericGraph

CallableGraph is a specific ConcreteGraph that includes a mapping for input Tensors to your graph. 

For example, say I have a callable graph `linear`. It has 3 inputs: `x`, `weight` and `bias`.

x is an input might be a different Tensor at each callsite. `weight` and `bias` I want to be the same at each callsite. Specifically they should be variables.
So we want calling `linear` to look like: `linear.call(x)` where the same `weight` and `bias` are passed to PopART everytime this is executed. 
To do this `linear` will have the following map: `{ subgraph_weight: weight, subgraph_bias: bias }`

So how do we construct `linear`. Lets start with a GenericGraph:
```
class Linear(GenericGraph):
    def build(self, x: pir.Tensor) -> pir.Tensor:
        self.weight = variable_def(np.random.normal((4, 4)).astype(pir.float32)
        self.bias = variable_def(np.zeros(4).astype(pir.float32))
        return (x @ self.weight) + self.bias
```
Here we have the definition of two variables `self.weight` and `self.bias`. Along with the execution of the graph: `(x @ self.weight) + self.bias`.
When `variable_def` is called we will have added two additional inputs to the graph.
In PopART, variables can only be added to the main_graph so to use these Tensors in the graph we'll have to pass them in a inputs.

Lets turn this into a `ConcreteGraph` so we can inspect the PopART graph.
```
graph = Linear().to_concrete(x)
graph.log_ir()
...
```
With this graph we can transform it, perhaps autodiff... 

Finally we want to call this graph. We intend to pass `x` in as a positional argument, however we need Tensors
to pass in for `self.weight` and `self.bias`. We use the definitions from earlier to construct variables for those inputs.
```
linear = graph.to_callable(create_variables=True)
print(linear)
...
```
Now we have all the parts to be able to execute this graph, we do that with:
```
linear.call(x)
```