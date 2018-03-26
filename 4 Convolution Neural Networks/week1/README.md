# Convolutional Neural Netowrks: Step by Step

In this assignment, you will implement convouutional(CONV) and pooling (POOL) layers in numpy, including both forward propagation and (optionally) backward propagation.

## Functions

- Convolution functions, including:
    - Zero Padding: Adds zeros around the border of an image
    - Single step of convolution: Takes an input volume; Applies a filte at every position of the input; Outputs another volume (usually of different size)
    - Convolution forward: This function takes as input A_prev, the activations output by the previous layer(for a batch of m inputs), F filters/weights denoted by W, and a bias vector denoted by b, where each filter has it own(single)bias.
    - Convolution backward
- Pooling functions, including:
    - Pooling forward: Implement the forward pass of the pooling layer.
    - Create mask: This function creates a "mask" matrix which keeps track of where the maximum of the matrix is.True(1) indicates the position of the maximum in X, the other entries are False(0).
    - Distribute value: Implement the function to equally distribute a value dz through a matrix of dimension shape.
    - Pooling backward: Implement the pool_backward function in both modes("max" and "average")

### Skills

- slice
    - To select a 2x2 slice at the upper left corner of a matrix "a_prev" (shape (5,5,3)), you would do:
    ```python
    a_slice_prev = a_prev[0:2, 0:2, :]
    ```
- Computing dA
    ```python
   da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
    ```
- Computing dW
    ```python
    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
    ```
- Computing db
    ```python
    db[:,:,:,c] += dZ[i, h, w, c]
    ```
-  Mask
   If you have a matrix X and a scalar x: `A = (X == x) ` will return a matrix A of the same size X such that:
   ```
   A[i,j] = True if X[i,j] = x
   A[i,j] = False if X[i,j] != x
   ```

- numpy funcitons
    - [np.pad](https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html)
    - [np.sum](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.sum.html)
    - np.zeros
    - np.multiply
    - np.max(), It computes the maximum of an array.
    - np.mean()
    - [np.ones()](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ones.html), Implement average pooling - backward pass

# Convolutional Neural Networks: Application

In this assigment, you will:

- Implement helper functions that you will use when implementing a TensorFlow model
- Implement a fully functioning ConvNet using TensorFlow

After this assigment you will be able to:

- Build and train a ConvNet in TensorFlow for a classification problem

## Functions

- Create placeholders: Implement the functiont to create placeholders for the input image X and the output Y. X should be of dimension[None,n_H0,n_W0,n_C0] and Y should be od dimension [None,n_y]
- Initialize parameters: Initialize weight/filters W1 and W2 using tf.contrib.layers.xavier_initializer(seed=0).To initalize a parameter W of shape[1,2,3,4] in Tensorflow,use:
```python
   W = tf.get_variable("W", [1,2,3,4], initializer = ...)
```
- Forward propagation: model:CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
- Compute cost
- Model: 
    - create placeholders
    - initialize parameters
    - forward propagate
    - computer the cost
    - create an optimizer


## Skills

- [tf.placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder)
- [tf.get_variable](https://www.tensorflow.org/api_docs/python/tf/get_variable): Gets an existing variable with these parameters or create a new one, to initalize  weight/filters

- [tf.contrib.layers.xavier_initializer(seed = 0)](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/xavier_initializer): Returns an initializer performing "Xavier" initialization for weights.

- In TensorFlow, There are built-in functions that carry out the convolution steps 
    -  [tf.nn.conv2d(X,W1,strides=[1,s,s,1],padding = 'SAME')](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d): given an input X and a group of filters W1, this function convolves W1's filters on X, The third input([1,f,f,1]) represents the strides for each dimension of the input(m, n_H_prev, n_W_prev, n_C_prev).
    - [tf.nn.max_pool(A, ksize = [1,f,f,1], strides = [1,s,s,1], padding='SAME')](https://www.tensorflow.org/api_docs/python/tf/nn/max_pool): given an input A, this function uses a window of size(f,f) and strides of size(s,s) to carry out max pooling over each window.
    - [tf.nn.relu(Z1)](https://www.tensorflow.org/api_docs/python/tf/nn/relu): computes the elementwise ReLU of Z1(which can be any shape).
    - [tf.contrib.layers.flatten(P)](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/flatten): given an input P, this function flattens each example into a 1D vector it while maintaining the batch-size. It returns a flattened tensor with shape[batch_size,k].
    - [tf.contrib.layers.fully_connected(F,num_outputs)](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/fully_connected): given a the flattened input F, it returns the output computed using a fully connected layer.
    - [tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels =Y)](https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits): computes the softmax entropy loss. This function both computes the softmax activation function as well as the resulting loss.
    - [tf.reduce_mean](https://www.tensorflow.org/api_docs/python/tf/reduce_mean): computes the mean of elements across dimensions of a tensor. Use this to sum the losses over all the examples to get the overall cost.


