# Face Recognition for the Happy House

- **Face Verification** - "is this the claimed person?". This is a 1:1 matching problem.
- **Face Recognition** - "who is this person?". This is a 1:K matching problem.

**In this assignment, you will:**
- Implement the triplet loss function
- Use a pretrained model to map face images into 128-dimensional encodings
- Use these encodings to perform face verification and face recognition

## Encoding face images into a 128-dimensional vector
### Using an ConvNet to compute encodings
- This network uses 96x96 dimensional RGB images as its input. Specifically, inputs a face image (or batch of $m$ face images) as a tensor of shape <img src="https://latex.codecogs.com/gif.latex?(m,_{n_{C}},_{n_{H}},_{n_{W}})=(m,3,96,96)">
- It outputs a matrix of shape <img src="https://latex.codecogs.com/gif.latex?(m,128)"> that encodes each input face image into a 128-dimensional vector

An encoding is a good one if:
- The encodings of two images of the same person are quite similar to each other
- The encodings of two images of different persons are very different

### The Triplet Loss
For an image <img src="https://latex.codecogs.com/gif.latex?x">, we denote its encoding <img src="https://latex.codecogs.com/gif.latex?f(x)">, where <img src="https://latex.codecogs.com/gif.latex?f"> is the function computed by the neural network.
Training will use triplets of images <img src="https://latex.codecogs.com/gif.latex?(A,P,N)">:
- <img src="https://latex.codecogs.com/gif.latex?A"> is an "Anchor" image--a picture of a person.
- <img src="https://latex.codecogs.com/gif.latex?P"> is a "Positive" image--a picture of the same person as the Anchor image.
- <img src="https://latex.codecogs.com/gif.latex?N"> is a "Negative" image--a picture of different person than the Anchor image.

<img src="https://latex.codecogs.com/gif.latex?$$\mid&space;\mid&space;f(A^{(i)})&space;-&space;f(P^{(i)})&space;\mid&space;\mid_2^2&space;&plus;&space;\alpha&space;<&space;\mid&space;\mid&space;f(A^{(i)})&space;-&space;f(N^{(i)})&space;\mid&space;\mid_2^2$$">

"Triplet cost":

<img src="https://latex.codecogs.com/gif.latex?$$\mathcal{J}&space;=&space;\sum^{m}_{i=1}&space;\large[&space;\small&space;\underbrace{\mid&space;\mid&space;f(A^{(i)})&space;-&space;f(P^{(i)})&space;\mid&space;\mid_2^2}_\text{(1)}&space;-&space;\underbrace{\mid&space;\mid&space;f(A^{(i)})&space;-&space;f(N^{(i)})&space;\mid&space;\mid_2^2}_\text{(2)}&space;&plus;&space;\alpha&space;\large&space;]&space;\small_&plus;&space;\tag{3}$$">

Here, we are using the notation "<img src="https://latex.codecogs.com/gif.latex?$[z]_&plus;$">" to denote <img src="https://latex.codecogs.com/gif.latex?$max(z,0)$">.  

- The term (1) is the squared distance between the anchor "A" and the positive "P" for a given triplet; you want this to be small. 
- The term (2) is the squared distance between the anchor "A" and the negative "N" for a given triplet, you want this to be relatively large, so it thus makes sense to have a minus sign preceding it. 
- <img src="https://latex.codecogs.com/gif.latex?$\alpha$"> is called the margin. It is a hyperparameter that you should pick manually. We will use <img src="https://latex.codecogs.com/gif.latex?$\alpha = 0.2$">. 

### Face Verification
Three steps:
1. Compute the encoding of the image from image_path
2. Compute the distance about this encoding and the encoding of the identity image stored in the database
3. Open the door if the distance is less than 0.7, else do not open.

As presented above, you should use the L2 distance (np.linalg.norm). (Note: In this implementation, compare the L2 distance, not the square of the L2 distance, to the threshold 0.7.) 

### Face recognition

Two Steps:

1. Compute the target encoding of the image from image_path
2. Find the encoding from the database that has smallest distance with the target encoding. 
    - Initialize the `min_dist` variable to a large enough number (100). It will help you keep track of what is the closest encoding to the input's encoding.
    - Loop over the database dictionary's names and encodings. To loop use `for (name, db_enc) in database.items()`.
        - Compute L2 distance between the target "encoding" and the current "encoding" from the database.
        - If this distance is less than the min_dist, then set min_dist to dist, and identity to name.

### Useful Functions
- `np.linalg.norm`
- `tf.reduce_sum()`
- `tf.square()`
- `tf.subtract()`
- `tf.add()`
- `tf.maximum`

## Summary
- Face verification solves an easier 1:1 matching problem; face recognition addresses a harder 1:K matching problem. 
- The triplet loss is an effective loss function for training a neural network to learn an encoding of a face image.
- The same encoding can be used for verification and recognition. Measuring distances between two images' encodings allows you to determine whether they are pictures of the same person. 