# Deep Learning & Art: Neural Style Transfer

**In this assigment, you will:**
- Implement the neural style transfer algorithm
- Generate novel artistic images using your algorithm


## Neural Style Transfer

**Three steps**
- Build the content cost function <img src="https://latex.codecogs.com/gif.latex?_{J_{content}}(C,G)">
- Build the style cost function <img src="https://latex.codecogs.com/gif.latex?_{J_{style}}(S,G)">
- Put it together to get <img src="https://latex.codecogs.com/gif.latex?J(G)=\alpha&space;_{J_{content}}(C,G)&plus;_\beta&space;{J_{style}}(S,G)">

### Computing the content cost
***Three steps:***
1. Retrieve dimensions from a_G:
    - To retrieve dimensions from a tensor X, use: X.get_shape().as_list()
2. Unroll a_C and a_G as explained in the picture above
    
3. Compute the content cost:
    

### Style matrix(Gram matrix)

The gram matrix of A is <img src="https://latex.codecogs.com/gif.latex?^{_{G_{A}}}&space;=&space;A^{A^{T}}">.

### Style cost
**Three steps**
1. Retrieve dimensions from the hidden layer activations a_G: 
    - To retrieve dimensions from a tensor X, use: `X.get_shape().as_list()`
2. Unroll the hidden layer activations a_S and a_G into 2D matrices, as explained in the picture above.  
3. Compute the Style matrix of the images S and G. 
4. Compute the Style cost:
    
### Neural Style Transfer
1. Create an Interactive Session 
2. Load the content image 
3. Load the style image
4. Randomly initialize the image to be generated 
5. Load the VGG16 model
7. Build the TensorFlow graph:
    - Run the content image through the VGG16 model and compute the content cost
    - Run the style image through the VGG16 model and compute the style cost
    - Compute the total cost
    - Define the optimizer and the learning rate
8. Initialize the TensorFlow graph and run it for a large number of iterations, updating the generated image at every step.

## Function

- [tf.assign](https://www.tensorflow.org/api_docs/python/tf/assign): Feed the image to the model
- [tf.matmul](https://www.tensorflow.org/api_docs/python/tf/matmul) 
- [tf.transpose](https://www.tensorflow.org/api_docs/python/tf/transpose)
- [tf.transpose](https://www.tensorflow.org/versions/r1.3/api_docs/python/tf/transpose)
- [tf.reshape](https://www.tensorflow.org/versions/r1.2/api_docs/python/tf/reshape)
- [tf.reduce_sum](https://www.tensorflow.org/api_docs/python/tf/reduce_sum)
- [tf.square](https://www.tensorflow.org/api_docs/python/tf/square) 
- [tf.substract](https://www.tensorflow.org/api_docs/python/tf/subtract)

- tf.InteractiveSession: interactive session
- [scipy.misc.imread](https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.imread.html): load the image
- reshape_and_normalize_image: reshape and normalize the image

## Summary

- The content cost takes a hidden layer activation of the neural network, and measures how different <img src="https://latex.codecogs.com/gif.latex?^{a^{(C)}}"> and <img src="https://latex.codecogs.com/gif.latex?^{a^{(G)}}"> are.
- When we minimize the content cost later, this will help make sure G has similar content as C.
- The style of an image can be represented using the Gram matrix of a hidden layer's activations. However, we get even better results combining this representation from multiple different layers. This is in contrast to the content representation, where usually using just a single hidden layer is sufficient.
- Minimizing the style cost will cause the image G to follow the style of the image S.
- The total cost is a linear combination of the content cost <img src="https://latex.codecogs.com/gif.latex?^{J_{content}}(C,G)"> and the style cost <img src="https://latex.codecogs.com/gif.latex?^{J_{style}}(S,G)">
- <img src="https://latex.codecogs.com/gif.latex?\alpha"> and <img src="https://latex.codecogs.com/gif.latex?\beta"> are hyperparameters that control the relative weighting between content and style.
***
- Neural Style Transfer is an algorithm that given a content image C and a style image S can generate an artistic image
- It uses representations (hidden layer activations) based on a pretrained ConvNet. 
- The content cost function is computed using one hidden layer's activations.
- The style cost function for one layer is computed using the Gram matrix of that layer's activations. The overall style cost function is obtained using several hidden layers.
- Optimizing the total cost function results in synthesizing new images. 