# Building your Recurrent Neural Network - Step by Step

## Forward propagation for the basic Recurrent Neural Network

Implement an RNN:

**Steps:**
1. Implement the calculations needed for one time-step of the RNN.
2. Implement a loop over <img src="https://latex.codecogs.com/gif.latex?$T_x$ ">  time-steps in order to process all the inputs, one at a time.

### RNN cell
1. Compute the hidden state with tanh activation: <img src="https://latex.codecogs.com/gif.latex?$a^{\langle t \rangle} = \tanh(W_{aa} a^{\langle t-1 \rangle} + W_{ax} x^{\langle t \rangle} + b_a)$">.
2. Using your new hidden state <img src="https://latex.codecogs.com/gif.latex?$a^{\langle t \rangle}$">, compute the prediction <img src="https://latex.codecogs.com/gif.latex?$\hat{y}^{\langle t \rangle} = softmax(W_{ya} a^{\langle t \rangle} + b_y)$">. We provided you a function: `softmax`.
3. Store <img src="https://latex.codecogs.com/gif.latex?$(a^{\langle t \rangle}, a^{\langle t-1 \rangle}, x^{\langle t \rangle}, parameters)$"> in cache
4. Return <img src="https://latex.codecogs.com/gif.latex?$a^{\langle t \rangle}$"> , <img src="https://latex.codecogs.com/gif.latex?$y^{\langle t \rangle}$"> and cache

### RNN forward pass

**Instructions**:
1. Create a vector of zeros (<img src="https://latex.codecogs.com/gif.latex?$a$">) that will store all the hidden states computed by the RNN.
2. Initialize the "next" hidden state as <img src="https://latex.codecogs.com/gif.latex?$a_0$"> (initial hidden state).
3. Start looping over each time step, your incremental index is <img src="https://latex.codecogs.com/gif.latex?$t$"> :
    - Update the "next" hidden state and the cache by running `rnn_cell_forward`
    - Store the "next" hidden state in <img src="https://latex.codecogs.com/gif.latex?$a$"> (<img src="https://latex.codecogs.com/gif.latex?$t^{th}$"> position) 
    - Store the prediction in y
    - Add the cache to the list of caches
4. Return <img src="https://latex.codecogs.com/gif.latex?$a$">, <img src="https://latex.codecogs.com/gif.latex?$y$"> and caches

## Long Short-Term Memory (LSTM) network

### About the gates

#### - Forget gate

<img src="https://latex.codecogs.com/gif.latex?$$\Gamma_f^{\langle t \rangle} = \sigma(W_f[a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_f)\tag{1} $$">

#### - Update gate

<img src="https://latex.codecogs.com/gif.latex?$$\Gamma_u^{\langle t \rangle} = \sigma(W_u[a^{\langle t-1 \rangle}, x^{\{t\}}] + b_u)\tag{2} $$"> 

#### - Updating the cell

<img src="https://latex.codecogs.com/gif.latex?$$ \tilde{c}^{\langle t \rangle} = \tanh(W_c[a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_c)\tag{3} $$">

<img src="https://latex.codecogs.com/gif.latex?$$ c^{\langle t \rangle} = \Gamma_f^{\langle t \rangle}* c^{\langle t-1 \rangle} + \Gamma_u^{\langle t \rangle} *\tilde{c}^{\langle t \rangle} \tag{4} $$">

#### - Output gate

<img src="https://latex.codecogs.com/gif.latex?$$ \Gamma_o^{\langle t \rangle}=  \sigma(W_o[a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_o)\tag{5}$$">

<img src="https://latex.codecogs.com/gif.latex?$$ a^{\langle t \rangle} = \Gamma_o^{\langle t \rangle}* \tanh(c^{\langle t \rangle})\tag{6} $$">

## Useful Fuctions
- [numpy.matmul](https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html): Matrix product of two arrays
- [numpy.multiply](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.multiply.html): Multiply arguments element-wise
- numpy.tanh
- 
