# Character level language model - Dinosaurus land

**Target**
- How to store text data for processing using an RNN
- How to synthesize data, by sampling predictions at each time step and passing it to the next RNN-cell unit
- How to build a character-level text generation recurrent neural network
- Why clipping the gradients is important

## Building blocks of the model
- Gradient clipping: to avoid exploding gradients
- Sampling: a technique used to generate characters

### Gradient clipping

We will use a simple element-wise clipping procedure, in which every element of the gradient vector is clipped to lie between some range [-N,N].

### Sampling

**4 Steps**
- **Step 1**: Pass the network the first "dummy" input <img src="https://latex.codecogs.com/gif.latex?$x^{\langle&space;1&space;\rangle}&space;=&space;\vec{0}$"> (the vector of zeros). This is the default input before we've generated any characters. We also set <img src = "https://latex.codecogs.com/gif.latex?$a^{\langle&space;0&space;\rangle}&space;=&space;\vec{0}$">

- **Step 2**: Run one step of forward propagation to get <img src="https://latex.codecogs.com/gif.latex?$a^{\langle&space;1&space;\rangle}$"> and <img src="https://latex.codecogs.com/gif.latex?\hat{y}^{\langle1\rangle}">. Here are the equations:

<img src="https://latex.codecogs.com/gif.latex?$$&space;a^{\langle&space;t&plus;1&space;\rangle}&space;=&space;\tanh(W_{ax}&space;x^{\langle&space;t&space;\rangle&space;}&space;&plus;&space;W_{aa}&space;a^{\langle&space;t&space;\rangle&space;}&space;&plus;&space;b)\tag{1}$$">


<img src="https://latex.codecogs.com/gif.latex?$$&space;z^{\langle&space;t&space;&plus;&space;1&space;\rangle&space;}&space;=&space;W_{ya}&space;a^{\langle&space;t&space;&plus;&space;1&space;\rangle&space;}&space;&plus;&space;b_y&space;\tag{2}$$">

<img src="https://latex.codecogs.com/gif.latex?$$&space;\hat{y}^{\langle&space;t&plus;1&space;\rangle&space;}&space;=&space;softmax(z^{\langle&space;t&space;&plus;&space;1&space;\rangle&space;})\tag{3}$$">

Note that <img src="https://latex.codecogs.com/gif.latex?\hat{y}^{\langle&space;t&plus;1&space;\rangle&space;}"> is a (softmax) probability vector (its entries are between 0 and 1 and sum to 1). <img src="https://latex.codecogs.com/gif.latex?\hat{y}^{\langle&space;t&plus;1&space;\rangle}_i"> represents the probability that the character indexed by "i" is the next character.  We have provided a `softmax()` function that you can use.

- **Step 3**: Carry out sampling: Pick the next character's index according to the probability distribution specified by <img src="https://latex.codecogs.com/gif.latex?\hat{y}^{\langle&space;t&plus;1&space;\rangle&space;}">. This means that if <img src="https://latex.codecogs.com/gif.latex?\hat{y}^{\langle&space;t&plus;1&space;\rangle&space;}_i&space;=&space;0.16">, you will pick the index "i" with 16% probability. To implement it, you can use [`np.random.choice`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.choice.html).

Here is an example of how to use `np.random.choice()`:
```python
np.random.seed(0)
p = np.array([0.1, 0.0, 0.7, 0.2])
index = np.random.choice([0, 1, 2, 3], p = p.ravel())
```
This means that you will pick the `index` according to the distribution: 
$P(index = 0) = 0.1, P(index = 1) = 0.0, P(index = 2) = 0.7, P(index = 3) = 0.2$.

- **Step 4**: The last step to implement in `sample()` is to overwrite the variable `x`, which currently stores <img src="https://latex.codecogs.com/gif.latex?x^{\langle&space;t&space;\rangle&space;}">, with the value of <img src="https://latex.codecogs.com/gif.latex?x^{\langle&space;t&space;&plus;&space;1&space;\rangle&space;}">. You will represent <img src="https://latex.codecogs.com/gif.latex?x^{\langle&space;t&space;&plus;&space;1&space;\rangle&space;}"> by creating a one-hot vector corresponding to the character you've chosen as your prediction. You will then forward propagate <img src="https://latex.codecogs.com/gif.latex?x^{\langle&space;t&space;&plus;&space;1&space;\rangle&space;}"> in Step 1 and keep repeating the process until you get a "\n" character, indicating you've reached the end of the dinosaur name. 

## Building the language model
### Gradient descent
In this section you will implement a function performing one step of stochastic gradient descent (with clipped gradients). You will go through the training examples one at a time, so the optimization algorithm will be stochastic gradient descent. As a reminder, here are the steps of a common optimization loop for an RNN:

- Forward propagate through the RNN to compute the loss
- Backward propagate through time to compute the gradients of the loss with respect to the parameters
- Clip the gradients if necessary
- Update your parameters using gradient descent

### Training the model


## Useful Functions
- [numpy.clip](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.clip.html): graduent clipping
- [`np.random.choice`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.choice.html): Carry out sampling