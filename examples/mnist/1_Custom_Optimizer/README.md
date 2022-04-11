# Custom Optimizer
We have seen in the [previous notebook]() that we can easily create graphs with an internal state using the ```addons.Module``` class. 
Besides layers, also optimizers often carry a state. Hence, they are a perfect use case for the ```Module``` class.

In this notebook we will show how to create a custom optimizer with state, the Adam optimizer.

```python
'''
Adam optimizer.
Defines adam update step for a single variable
'''
class Adam(addons.Module):
    # we need to specify in_sequence because a lot of operations are in place and their order 
    # shouldn't be rearranged 
    @popxl.in_sequence()
    def build(self,
              var: popxl.TensorByRef,
              grad: popxl.Tensor,
              *,
              lr: Union[float, popxl.Tensor],
              beta1: Union[float, popxl.Tensor] = 0.9,
              beta2: Union[float, popxl.Tensor] = 0.999,
              eps: Union[float, popxl.Tensor] = 1e-5,
              weight_decay: Union[float, popxl.Tensor] = 1e-2,
              first_order_dtype: popxl.dtype = popxl.float16,
              bias_correction: bool = True):

        # gradient estimators for the variable var - same shape as the variable 
        first_order = self.add_input_tensor("first_order", partial(np.zeros, var.shape), first_order_dtype, by_ref=True)
        ops.var_updates.accumulate_moving_average_(first_order, grad, f=beta1)

        # variance estimators for the variable var - same shape as the variable 
        second_order = self.add_input_tensor("second_order", partial(np.zeros, var.shape), popxl.float32, by_ref=True)
        ops.var_updates.accumulate_moving_average_square_(second_order, grad, f=beta2)

        # adam is a biased estimator: provide the step to correct bias
        step = None
        if bias_correction:
            step = self.add_input_tensor("step", partial(np.zeros, ()), popxl.float32, by_ref=True)

        # calculate the weight increment with adam euristic 
        updater = ops.var_updates.adam_updater(
            first_order, second_order,
            weight=var,
            weight_decay=weight_decay,
            time_step=step,
            beta1=beta1,
            beta2=beta2,
            epsilon=eps)

        # in place weight update: w += (-lr)*dw
        ops.scaled_add_(var, updater, b=-lr)
```
The first important thing to note is that all the build method is executed **in sequence**, as we've added the ```@popxl.in_sequence()``` decorator. 
It is necessary since most of the optimizer operations are **in place**, hence their order of execution must be stricly preserved.

Note also that the ```var``` input is a ```popxl.TensorByRef```: any change made to this variable will be automatically copied to the parent graph. See [TensorByRef]() for more information.

Allowing a ```Union[float, popxl.Tensor]``` type for the optimizer parameters such as the learning rate or the weight decay gives this module an interesting property.
If the parameter is provided as a ```float```, it will be "baked" into the graph, with no possibility of changing it.
Instead, if the parameter is a ```Tensor``` (or ```TensorSpec```) it will appear as an input to the graph, which needs to be provided when calling the graph. If you plan to change a parameter (for example, because you have a learning rate schedule), this is the way to go.

The rest of the logic is straightforward:

- we update the first moment, estimator for the gradient of the variable
- we update the second moment, estimator for the variance of the variable
- we optionally correct the estimators, since they are biased
- we compute the increment for the variable, ```dw```
- we update the variable ``` w += (-lr) * dw```

The ```ops.var_updates``` module contains several useful pre-made update rules, but you can also make your own. In this example we are using three of them:

- ```ops.var_updates.accumulate_moving_average_(average, new_sample, coefficient)``` updates ```average``` in place with an exponential moving average rule: 
    ```
    average = (coefficient * average) + ((1-coefficient) * new_sample)
    ```
- ```accumulate_moving_average_square_(average, new_sample, coefficient)``` does the same, but using the square of the sample.  
- ```ops.var_updates.adam_updater(...)``` returns the adam increment ```dw``` which is required for the weight update, computed using adam internal state, i.e. the first and second moments.

Let's inspect the optimizer graph and its use in a simple example.
The output of the example is shown below. Note the difference in the graph inputs for a float learning rate parameter or a tensor learning rate parameter. 

```python
ir = popxl.Ir()
ir.replication_factor = 1 

with ir.main_graph:
    var = popxl.variable(np.ones((2,2)),popxl.float32)
    grad = popxl.variable(np.full((2,2),0.1),popxl.float32)
    # create graph and factories - float learning rate
    adam_facts, adam = Adam().create_graph(var, var.spec, lr=1e-3)
    # create graph and factories - Tensor learning rate
    adam_facts_lr, adam_lr = Adam().create_graph(var, var.spec, lr=popxl.TensorSpec((),popxl.float32))
    print("Adam with float learning rate\n")
    print(adam.print_schedule())
    print("\n Adam with tensor learning rate\n")
    print(adam_lr.print_schedule())
    # instantiate optimizer variables 
    adam_state = adam_facts.init()
    adam_state_lr = adam_facts_lr.init()
    # optimization step for float lr: call the bound graph providing the variable to update and the gradient 
    adam.bind(adam_state).call(var, grad)
    # optimization step for tensor lr: call the bound graph providing the variable to update, the gradient and the learning rate
    adam_lr.bind(adam_state_lr).call(var, grad, popxl.constant(1e-3))

ir.num_host_transfers = 1
session = popxl.Session(ir,"ipu_hw")
print("\n Before adam update")
var_data = session.get_tensor_data(var)
state = session.get_tensors_data(adam_state.tensors)
print("Variable:\n", var)
print("Adam state:")
for name, data in state.items():
    print(name,'\n', state[name])

session.run()

print("\n After adam update")
var_data = session.get_tensor_data(var)
state = session.get_tensors_data(adam_state.tensors)
print("Variable:\n", var)
print("Adam state:")
for name, data in state.items():
    print(name,'\n', state[name])

session.device.detach()
```

**Adam with float learning rate**
```
Graph : Adam_subgraph(0)
  (%1, %2, first_order=%3, second_order=%4, step=%5) -> () {
    Accumulate.100 (%3 [(2, 2) float16], %2 [(2, 2) float32]) -> (%6 [(2, 2) float16])
    Accumulate.101 (%4 [(2, 2) float32], %2 [(2, 2) float32]) -> (%7 [(2, 2) float32])
    AdamUpdater.102 (%1 [(2, 2) float32], %3 [(2, 2) float16], %4 [(2, 2) float32], %5 [() float32]) -> (%8 [(2, 2) float32])
    ScaledAddLhsInplace.103 (%1 [(2, 2) float32], %8 [(2, 2) float32]) -> (%9 [(2, 2) float32])
  }
```

**Adam with tensor learning rate**
```
Graph : Adam_subgraph(1)
  (%1, %2, %3, first_order=%4, second_order=%5, step=%6) -> () {
    Accumulate.104 (%4 [(2, 2) float16], %2 [(2, 2) float32]) -> (%7 [(2, 2) float16])
    Accumulate.105 (%5 [(2, 2) float32], %2 [(2, 2) float32]) -> (%8 [(2, 2) float32])
    AdamUpdater.106 (%1 [(2, 2) float32], %4 [(2, 2) float16], %5 [(2, 2) float32], %6 [() float32]) -> (%9 [(2, 2) float32])
    Neg.107 (%3 [() float32]) -> (%10 [() float32])
    ScaledAddLhsInplace.108 (%1 [(2, 2) float32], %9 [(2, 2) float32], %10 [() float32]) -> (%11 [(2, 2) float32])
  }
```

**Before adam update**
```
Variable:
 Tensor[t popxl.dtypes.float32 (2, 2)]
Adam state:
Tensor[first_order popxl.dtypes.float16 (2, 2)] 
 [[0. 0.]
 [0. 0.]]
Tensor[second_order popxl.dtypes.float32 (2, 2)] 
 [[0. 0.]
 [0. 0.]]
Tensor[step popxl.dtypes.float32 ()] 
 0.0
```

**After adam update**
```
Variable:
 Tensor[t popxl.dtypes.float32 (2, 2)]
Adam state:
Tensor[first_order popxl.dtypes.float16 (2, 2)] 
 [[0.009995 0.009995]
 [0.009995 0.009995]]
Tensor[second_order popxl.dtypes.float32 (2, 2)] 
 [[9.9998715e-06 9.9998715e-06]
 [9.9998715e-06 9.9998715e-06]]
Tensor[step popxl.dtypes.float32 ()] 
 1.0
```

# Mnist with Adam
We can now refactor our mnist example to incorporate the Adam optimizer. 
Note that we need an optimizer for each variable: we first define a utility function to create all the graphs and perform a full weight update for all the variables in the neural network. 

We will use a float learning rate, since we don't plan to change its value during training.

The training code is almost unchanged from that of the previous tutorial, the only different piece is the code related to the optimizer in  ```train_program```. Also, since we are using Adam, we need to use a smaller learning rate. 

You will notice that we create the Adam module using 
```python
optimizer = Adam(cache=True)
```
Using `cache=True` will enable graph reuse, if possible, when calling `optimizer.create_graph`. For our optimizer this would be when there are multiple variables with the same shape/dtype.