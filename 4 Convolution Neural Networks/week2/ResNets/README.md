# Residual Networks

**In this assignment, you will:**
- Implement the basic building blocks of ResNets.
- Put together these building blocks to implement and train a state-of-the-art neural network for image classification

This assignment will be done in Keras.

## Five Steps to build the identity block:

First component of main path:
- The first CONV2D has <img src="https://latex.codecogs.com/gif.latex?_{F_{1}}"> filters of shape(1,1) and a stride of (s,s). Its padding is "valid" and its name should be conv_name_base + '2a'.
- The first BatchNorm is normalizing the channels axis. Its name should be bn_name_base + '2a'.
- Then apply the ReLU activation function. This has no name and no hyperparameters.

Second component of main path;
- The second CONV2D has <img src="https://latex.codecogs.com/gif.latex?_{F_{2}}"> filters of (f,f) and a stride of (1,1). Its padding is "same" and it's name should be conv_name_base + '2b'.
- The second BatchNorm is normalizing the channels axis. Its name should be bn_name_base + '2b'.
- Then apply the ReLU activation function. This has no name and no hyperparameters.

Third component of main path:
- The third CONV2D has <img src="https://latex.codecogs.com/gif.latex?_{F_{3}}"> filters of (1,1) and a stride of (1,1). Its padding is "vaild" and it's name should be conv_name_base + '2c'.
- The third BatchNorm is normalizing the channels axis. Its name should be bn_name_base + '2c'. Note that there is no ReLU activation function in this component.

Shortcut path:
- The CONV2D has <img src="https://latex.codecogs.com/gif.latex?_{F_{3}}"> filters of shape(1,1) and a stride of (s,s). It's padding is "vaild" and its name should be conv_name_base + 'l'.
- The BatchNorm is normalizing the channels axis. Its name should be bn_name_base + 'l'.

Final Step:
- The shortcut and the main path values are added together.
- Then apply the ReLU activation function. This has no name and no hyperparameters.

## ResNet-50 model

<img src="images/resnet_kiank.png" style="width:850px;height:150px;">

- Zero-padding pads the input with a pad of (3,3)
- Stage 1:
    - The 2D Convolution has 64 filters of shape (7,7) and uses a stride of (2,2). Its name is "conv1".
    - BatchNorm is applied to the channels axis of the input.
    - MaxPooling uses a (3,3) window and a (2,2) stride.
- Stage 2:
    - The convolutional block uses three set of filters of size [64,64,256], "f" is 3, "s" is 1 and the block is "a".
    - The 2 identity blocks use three set of filters of size [64,64,256], "f" is 3 and the blocks are "b" and "c".
- Stage 3:
    - The convolutional block uses three set of filters of size [128,128,512], "f" is 3, "s" is 2 and the block is "a".
    - The 3 identity blocks use three set of filters of size [128,128,512], "f" is 3 and the blocks are "b", "c" and "d".
- Stage 4:
    - The convolutional block uses three set of filters of size [256, 256, 1024], "f" is 3, "s" is 2 and the block is "a".
    - The 5 identity blocks use three set of filters of size [256, 256, 1024], "f" is 3 and the blocks are "b", "c", "d", "e" and "f".
- Stage 5:
    - The convolutional block uses three set of filters of size [512, 512, 2048], "f" is 3, "s" is 2 and the block is "a".
    - The 2 identity blocks use three set of filters of size [512, 512, 2048], "f" is 3 and the blocks are "b" and "c".
- The 2D Average Pooling uses a window of shape (2,2) and its name is "avg_pool".
- The flatten doesn't have any hyperparameters or name.
- The Fully Connected (Dense) layer reduces its input to the number of classes using a softmax activation. Its name should be `'fc' + str(classes)`.  


## Functions

- [Conv2D](https://keras.io/layers/convolutional/#conv2d)
- [BatchNorm](https://keras.io/layers/normalization/#batchnormalization)
- [Add](https://keras.io/layers/merge/#add)
- [Activation](https://keras.io/layers/core/#activation)
- [Average pooling](https://keras.io/layers/pooling/#averagepooling2d)
- [Max pooling](https://keras.io/layers/pooling/#maxpooling2d)
- [Fully connected layer](https://keras.io/layers/core/#dense)