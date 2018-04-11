# Keras tutorial - the Happy House

1. Learn to use Keras, a high-level neural networks API (programming framework), written in Python and capable of running on top of several lower-level frameworks including TensorFlow and CNTK. 
2. See how you can in a couple of hours build a deep learning algorithm.

## Building a model in Keras

**An example of a model in Keras:**

```python
def model(input_shape):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')
    
    return model
```
Note that Keras uses a different convention with variable names than we've previously used with numpy and TensorFlow. In particular, rather than creating and assigning a new variable on each step of forward propagation such as `X`, `Z1`, `A1`, `Z2`, `A2`, etc. for the computations for the different layers, in Keras code each line above just reassigns `X` to a new value using `X = ...`. In other words, during each step of forward propagation, we are just writing the latest value in the commputation into the same variable `X`. The only exception was `X_input`, which we kept separate and did not overwrite, since we needed it at the end to create the Keras model instance (`model = Model(inputs = X_input, ...)` above). 

**To train and test the model, there are four steps in Keras:**
1. Create the model by calling the function above
2. Compile the model by calling `model.compile(optimizer = "...", loss = "...", metrics = ["accuracy"])`
3. Train the model on train data by calling `model.fit(x = ..., y = ..., epochs = ..., batch_size = ...)`. Note that if you run fit() again, the model will continue to train with the parameters it has already learnt instead of reinitializing them
4. Test the model on test data by calling `model.evaluate(x = ..., y = ...)`

Example:

```python
# Create the model
happyModel = HappyModel((64,64,3)

# Compile the model
# compile the model to configure the learning process
happyModel.compile(optimizer=keras.optimizers.Adam(lr=0.001,beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss = "binary_crossentropy", metrics = ["accuracy"])

# Train the model on train data
#  train the model. Choose the number of epochs and the batch size.
#  Note that if you run fit() again, the model will continue to train with the parameters it has already learnt instead of reinitializing them.
happyModel.fit(X_train, Y_train, epochs=40, batch_size= 12)

# Test the model on test data
# test/evaluate the model
preds = happyModel.evaluate(x=X_test,y=Y_test)
### END CODE HERE ###
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

```

If you have not yet achieved a very good accuracy (let's say more than 80%), here're some things you can play around with to try to achieve it:

- Try using blocks of CONV->BATCHNORM->RELU such as:
```python
X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv0')(X)
X = BatchNormalization(axis = 3, name = 'bn0')(X)
X = Activation('relu')(X)
```
until your height and width dimensions are quite low and your number of channels quite large (≈32 for example). You are encoding useful information in a volume with a lot of channels. You can then flatten the volume and use a fully-connected layer.
- You can use MAXPOOL after such blocks. It will help you lower the dimension in height and width.
- Change your optimizer. We find Adam works well. 
- If the model is struggling to run and you get memory issues, lower your batch_size (12 is usually a good compromise)
- Run on more epochs, until you see the train accuracy plateauing. 

Even if you have achieved a good accuracy, please feel free to keep playing with your model to try to get even better results. 


## Useful Keras Functions

- [Activation](https://keras.io/layers/core/#activation): Applies an activation function to an output. [Available activations](https://keras.io/activations/#available-activations): `softmax`, `relu`, `tanh` ...
- [BatchNormalization](https://keras.io/layers/normalization/#batchnormalization): Batch normalization layer (Ioffe and Szegedy, 2014). 

  Normalize the activations of the previous layer at each batch, i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.
- [Conv2D](https://keras.io/layers/convolutional/#conv2d)
- [ZeroPadding2D](https://keras.io/layers/convolutional/#zeropadding2d)
- [Flatten](https://keras.io/layers/core/#flatten)
- [Dense](https://keras.io/layers/core/#dense)

- [AveragePooling2D()](https://keras.io/layers/pooling/#averagepooling2d): Average pooling operation for spatial data.
- [GlobalMaxPooling2D()](https://keras.io/layers/pooling/#globalmaxpooling2d): Global max pooling operation for spatial data.
- [Dropout()](https://keras.io/layers/core/#dropout): Applies Dropout to the input.
- [model.compile()](https://keras.io/models/model/): Configures the model for training.
- [model.fit()](https://keras.io/models/model/): Trains the model for a fixed number of epochs (iterations on a dataset).
- [model.evaluate()](https://keras.io/models/model/): Returns the loss value & metrics values for the model in test mode. Computation is done in batches.
- `model.summary()`: prints the details of your layers in a table with the sizes of its inputs/outputs
- [plot_model()](https://keras.io/visualization/): plots your graph in a nice layout. You can even save it as ".png" using SVG() if you'd like to share it on social media ;). It is saved in "File" then "Open..." in the upper bar of the notebook.


## Summary

- Keras is a tool we recommend for rapid prototyping. It allows you to quickly try out different model architectures. Are there any applications of deep learning to your daily life that you'd like to implement using Keras? 
- Remember how to code a model in Keras and the four steps leading to the evaluation of your model on the test set. Create->Compile->Fit/Train->Evaluate/Test.
