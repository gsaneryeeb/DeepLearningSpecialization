# Building your Recurrent Neural Network - Step by Step

## Forward propagation for the basic Recurrent Neural Network

Implement an RNN:

**Steps:**
1. Implement the calculations needed for one time-step of the RNN.
2. Implement a loop over <img src="https://latex.codecogs.com/gif.latex?$T_x$">  time-steps in order to process all the inputs, one at a time.

### RNN cell
1. Compute the hidden state with tanh activation: <img src="https://latex.codecogs.com/gif.latex?$a^{\langle&space;t&space;\rangle}&space;=&space;\tanh(W_{aa}&space;a^{\langle&space;t-1&space;\rangle}&space;&plus;&space;W_{ax}&space;x^{\langle&space;t&space;\rangle}&space;&plus;&space;b_a)$">.
2. Using your new hidden state <img src="https://latex.codecogs.com/gif.latex?$a^{\langle&space;t&space;\rangle}$">, compute the prediction <img src="https://latex.codecogs.com/gif.latex?\hat{y}^{\langle&space;t&space;\rangle}&space;=&space;softmax(W_{ya}&space;a^{\langle&space;t&space;\rangle}&space;&plus;&space;b_y)">. We provided you a function: `softmax`.
3. Store <img src="https://latex.codecogs.com/gif.latex?$(a^{\langle&space;t&space;\rangle},&space;a^{\langle&space;t-1&space;\rangle},&space;x^{\langle&space;t&space;\rangle},&space;parameters)$"> in cache
4. Return <img src="https://latex.codecogs.com/gif.latex?$a^{\langle&space;t&space;\rangle}$&space;,&space;$y^{\langle&space;t&space;\rangle}$"> and cache

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

<img src="https://latex.codecogs.com/gif.latex?$$\Gamma_f^{\langle&space;t&space;\rangle}&space;=&space;\sigma(W_f[a^{\langle&space;t-1&space;\rangle},&space;x^{\langle&space;t&space;\rangle}]&space;&plus;&space;b_f)\tag{1}&space;$$">

#### - Update gate

<img src="https://latex.codecogs.com/gif.latex?$$\Gamma_u^{\langle&space;t&space;\rangle}&space;=&space;\sigma(W_u[a^{\langle&space;t-1&space;\rangle},&space;x^{\{t\}}]&space;&plus;&space;b_u)\tag{2}&space;$$"> 

#### - Updating the cell

<img src="https://latex.codecogs.com/gif.latex?$$&space;\tilde{c}^{\langle&space;t&space;\rangle}&space;=&space;\tanh(W_c[a^{\langle&space;t-1&space;\rangle},&space;x^{\langle&space;t&space;\rangle}]&space;&plus;&space;b_c)\tag{3}&space;$$">



<img src="https://latex.codecogs.com/gif.latex?$$&space;c^{\langle&space;t&space;\rangle}&space;=&space;\Gamma_f^{\langle&space;t&space;\rangle}*&space;c^{\langle&space;t-1&space;\rangle}&space;&plus;&space;\Gamma_u^{\langle&space;t&space;\rangle}&space;*\tilde{c}^{\langle&space;t&space;\rangle}&space;\tag{4}&space;$$">

#### - Output gate

<img src="https://latex.codecogs.com/gif.latex?$$&space;\Gamma_o^{\langle&space;t&space;\rangle}=&space;\sigma(W_o[a^{\langle&space;t-1&space;\rangle},&space;x^{\langle&space;t&space;\rangle}]&space;&plus;&space;b_o)\tag{5}$$">

<img src="https://latex.codecogs.com/gif.latex?$$&space;a^{\langle&space;t&space;\rangle}&space;=&space;\Gamma_o^{\langle&space;t&space;\rangle}*&space;\tanh(c^{\langle&space;t&space;\rangle})\tag{6}&space;$$">

## Useful Fuctions
- [numpy.matmul](https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html): Matrix product of two arrays
- [numpy.multiply](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.multiply.html): Multiply arguments element-wise
- numpy.tanh
- 
