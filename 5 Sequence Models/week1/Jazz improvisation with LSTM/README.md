# Improvise a Jazz Solo with an LSTM Network (Keras)
**Goal**
- Apply an LSTM to music gereration.
- Generate your own jazz music with deep learning.

Building the model
In this part you will build and train a model that will learn musical patterns. To do so, you will need to build a model that takes in X of shape <img src="https://latex.codecogs.com/gif.latex?(m,&space;T_x,&space;78)"> and Y of shape <img src="https://latex.codecogs.com/gif.latex?(T_y,&space;m,&space;78)">. We will use an LSTM with 64 dimensional hidden states. Lets set `n_a = 64`. 

Implement `djmodel()`. You will need to carry out 2 steps:
1. Create an empty list "outputs" to save the outputs of the LSTM Cell at every time step.
2. Loop for <img src="https://latex.codecogs.com/gif.latex?t&space;\in&space;1,&space;\ldots,&space;T_x">:

    A. Select the "t"th time step vector from X. The shape of this selection should be (78,). To do so, create a custom [Lambda](https://keras.io/layers/core/#lambda) layer in Keras by using this line of code:
```
    x = Lambda(lambda x: X[:,t,:])(X)
```
It is creating a "temporary" or "unnamed" function (that's what Lambda function are) that extracts out the appropriate one-hot vector, and making this function a Keras `Layer` object to apply to `X`.
    B. Reshape x to be (1,78). You may find the `reshapor()` layer (defined below) helpful. 
    C. Run x through one step of LSTM_cell. Remember to initialize the LSTM_cell with the previous step's hidden state <img src="https://latex.codecogs.com/gif.latex?a"> and cell state <img src="https://latex.codecogs.com/gif.latex?c">. Use the following formatting:
```python
    a, _, c = LSTM_cell(input_x, initial_state=[previous hidden state, previous cell state])
```    

## Useful Functions
**Keras:**
- [Reshape()](https://keras.io/layers/core/#reshape): Reshapes an output to a certain shape.
- [LSTM()](https://keras.io/layers/recurrent/#lstm): Long Short-Term Memory layer - Hochreiter 1997. 
- [Dense()](https://keras.io/layers/core/#dense): Just your regular densely-connected NN layer.