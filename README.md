# Simple Neural Network Maker
neural network maker uses preceptrons to create a neural network and implements forward and back propagation procedure using Chain rule instead of Matrix-based approach

## Create Network:
### Create Perceptron:
a perceptron apply activation function on it's inputs sum.
use following command to create a perceptron `p` with initial weights `W` and activation function `f`:
```
p = Perceptron(W,f)
```
to calculate perceptron output and its derivatives with input `X` can use this command:
```
p.cal_output(X)
```
and for updating it's weights after back-propagation:
```
p.update_weights()
```
### Build Network:
to create a neural network with loss function `loss`
```
n = Network(loss)
```
a neural network must have at least one 'input' and one 'output' perceptron.
you can use several 'hidden' neurons in your network.
in designing the structure of network there is almost no limit. you can connect neurons from inputs to output in any way.
for adding perceptron `p` with `i` input and type `t`(output,input,hidden) to the network use:
```
n.add_node(p,t,i)
```
and for connecting neuron `p1` to `i`'th `p2`'s input use this
```
n.connect_nodes(p1,p2,i)
```
after designing the network, it's time to learn our network with training dataset. as you know, training procedure contains two part: forward-propagation and back-propagation.
in each epoch, we once do forward-propagation and then back-propagation.
for feeding data to network we do forward-propagation. with training data `X`, use following command:
```
n.forward_prop(X)
```
for updating weights we do back-propagation. with training data labels `Y` and learning rate `alpha`, use following command:
```
n.back_prop(alpha,Y)
```

## To-Do
- [ ] add `compile` function to `Network` class that checks connections between nodes and their inputs
- [ ] add `Leyer` class for fully-connected networks
- [ ] make a higher level api

