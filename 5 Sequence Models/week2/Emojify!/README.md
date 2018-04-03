# Emojify!

In this exercise, you'll start with a baseline model (Emojifier-V1) using word embeddings, then build a more sophisticated model (Emojifier-V2) that further incorporates an LSTM.

## Baseline model: Emojifier-V1
### Implementing Emojifier-V1
Implement `sentence_to_avg()` funciton:
1. Convert every senetence to lower-case, then split the sentence into a list of words. `X.lower()` and `X.split()` might be useful.
2. For each word in the sentence, access its GloVe representation. Then, average all these values.

### Model
Implement `model()` function: Assuming here that $Yoh$ ("Y one hot") is the one-hot encoding of the output labels, the equations you need to implement in the forward pass and to compute the cross-entropy cost are:

<img src="https://latex.codecogs.com/gif.latex?z^{(i)}&space;=&space;W&space;.&space;avg^{(i)}&space;&plus;&space;b">

<img src="https://latex.codecogs.com/gif.latex?a^{(i)}&space;=&space;softmax(z^{(i)})">

<img src="https://latex.codecogs.com/gif.latex?\mathcal{L}^{(i)}&space;=&space;-&space;\sum_{k&space;=&space;0}^{n_y&space;-&space;1}&space;Yoh^{(i)}_k&space;*&space;log(a^{(i)}_k)">

## Emojifier-V2: Using LSTMs in Keras:

### Overview of the model

Here is the Emojifier-v2 you will implement:

<img src="images/emojifier-v2.png" style="width:700px;height:400px;"> 

### Keras and mini-batching

In this exercise, we want to train Keras using mini-batches. However, most deep learning frameworks require that all sequences in the same mini-batch have the same length. This is what allows vectorization to work: If you had a 3-word sentence and a 4-word sentence, then the computations needed for them are different (one takes 3 steps of an LSTM, one takes 4 steps) so it's just not possible to do them both at the same time.

The common solution to this is to use padding. Specifically, set a maximum sequence length, and pad all sequences to the same length. For example, of the maximum sequence length is 20, we could pad every sentence with "0"s so that each input sentence is of length 20. Thus, a sentence "i love you" would be represented as <img src="https://latex.codecogs.com/gif.latex?(e_{i},&space;e_{love},&space;e_{you},&space;\vec{0},&space;\vec{0},&space;\ldots,&space;\vec{0})">. In this example, any sentences longer than 20 words would have to be truncated. One simple way to choose the maximum sequence length is to just pick the length of the longest sentence in the training set. 

### The Embedding layer

The `Embedding()` layer takes an integer matrix of size (batch size, max input length) as input. This corresponds to sentences converted into lists of indices (integers), as shown in the figure below.

<img src="images/embedding1.png" style="width:700px;height:250px;">
<caption><center> Embedding layer. This example shows the propagation of two examples through the embedding layer. Both have been zero-padded to a length of `max_len=5`. The final dimension of the representation is  `(2,max_len,50)` because the word embeddings we are using are 50 dimensional. </center></caption>

The largest integer (i.e. word index) in the input should be no larger than the vocabulary size. The layer outputs an array of shape (batch size, max input length, dimension of word vectors).

The first step is to convert all your training sentences into lists of indices, and then zero-pad all these lists so that their length is the length of the longest sentence. 

Implement `pretrained_embedding_layer()`:
1. Initialize the embedding matrix as a numpy array of zeros with the correct shape.
2. Fill in the embedding matrix with all the word embedding extracted from `word_to_vec_map`.
3. Define Keras embedding layer. Use [Embedding()](https://keras.io/layers/embeddings/). Be sure to make this layer non-trainable, by setting `trainable = False` when calling `Embedding()`. If you were to set trainable = True, then it will allow the optimization algorithm to modify the values of the word embeddings.
4. Set the embedding weights to be equal to the embedding matrix.

### Building the Emojifier-V2

Lets now build the Emojifier-V2 model. You will do so using the embedding layer you have built, and feed its output to an LSTM network. 

<img src="images/emojifier-v2.png" style="width:700px;height:400px;"> <br>
<caption><center> **Figure 3**: Emojifier-v2. A 2-layer LSTM sequence classifier. </center></caption>

Implement `Emojify_V2()`, which builds a Keras graph of the architecture shown in Figure 3. The model takes as input an array of sentences of shape (`m`, `max_len`, ) defined by `input_shape`. It should output a softmax probability vector of shape (`m`, `C = 5`). You may need `Input(shape = ..., dtype = '...')`, [LSTM()](https://keras.io/layers/recurrent/#lstm), [Dropout()](https://keras.io/layers/core/#dropout), [Dense()](https://keras.io/layers/core/#dense), and [Activation()](https://keras.io/activations/).


## Useful Functions

- Confusion martix `plot_confusion_matrix()`: Printing the confusion matrix can also help understand which classes are more difficult for your model. A confusion matrix shows how often an example whose label is one class ("actual" class) is mislabeled by the algorithm with a different class ("predicted" class).

**Keras:**
- [Embedding()](https://keras.io/layers/embeddings/)
- [LSTM()](https://keras.io/layers/recurrent/#lstm)
- [Dropout()](https://keras.io/layers/core/#dropout)
- [Dense()](https://keras.io/layers/core/#dense)
- [Activation()](https://keras.io/activations/)


## Summary

- Even with a 127 training examples, you can get reasonably good model for Emojifying. This is due to the generalization power word vectors gives you.
- Emojify-V1 will perform poorly on sentences such as "This movie is not good and not enjoyable" beacause it doesn't understand combinations of words--it just averages all the words' embedding vectors together, without paying attention to the ordering of words. You will build a better algorithm in the next part.

- If you have an NLP task where the training set is small, using word embeddings can help your algorithm significantly. Word embeddings allow your model to work on words in the test set that may not even have appeared in your training set.
- Training sequence models in Keras (and in most other deep learning frameworks) requires a few important details:
    - To use mini-batches. the sequences need to be padded so that all the examples in a mini-batch have the same length.
    - An `Embedding()` layer can be initialized with pretrained values. These values can be either fixed or trained further on your dataset. If however your labeled dataset is small, it's usually not worth trying to train a large pre-trained set of embeddings.
   -  `LSTM()` has a flag called `return_sequences` to decide if you would like to return every hidden states or only the last one.
   - You can use `Dropout()` right after `LSTM()` to regularize your network.