# Trigger Word Detection

**Key of this assignment**
- Structure a speech recognition project
- Synthesize and process audion recordings to create train/dev datasets
- Train a trigger word detection model and make predictions

## Data synthesis: Creating a speech dataset

### From audio recordings to spectrograms

We will use audio sampled at 44100Hz(or 44100Hertz). This means the microphone gives us 44100 numbers per second. Thus, a 10 second audio clip is represented by 441000 numbers(=<img src="https://latex.codecogs.com/gif.latex?10&space;\times&space;44100>")

In order to help your sequence model more easily learn to detect triggerwords, we will compute a *spectrogram* of th audio. The spectrogtam tells us how much different frequencies are present in an audio clip at a moment in time.

If you've ever taken an advanced class on signal processing or on Fourier transforms, a spectrogram is computed by sliding a window over the raw audio signal, and calculates the most active frequencies in each window using a Fourier transform.

Note that even with 10 seconds being our default training example length, 10 seconds of time can be discretized to different numbers of value. You've seen 441000 (raw audio) and 5511 (spectrogram). In the former case, each step represents <img src="https://latex.codecogs.com/gif.latex?10/441000&space;\approx&space;0.000023"> seconds. In the second case, each step represents <img src="https://latex.codecogs.com/gif.latex?10/5511&space;\approx&space;0.0018"> seconds. 

For the 10sec of audio, the key values you will see in this assignment are:

- <img src="https://latex.codecogs.com/gif.latex?441000"> (raw audio)
- <img src="https://latex.codecogs.com/gif.latex?5511&space;=&space;T_x"> (spectrogram output, and dimension of input to the neural network). 
- <img src="https://latex.codecogs.com/gif.latex?10000"> (used by the `pydub` module to synthesize audio) 
- <img src="https://latex.codecogs.com/gif.latex?1375&space;=&space;T_y"> (the number of steps in the output of the GRU you'll build). 

Note that each of these representations correspond to exactly 10 seconds of time. It's just that they are discretizing them to different degrees. All of these are hyperparameters and can be changed (except the 441000, which is a function of the microphone). We have chosen values that are within the standard ranges uses for speech systems. 

Consider the <img src="https://latex.codecogs.com/gif.latex?T_y&space;=&space;1375"> number above. This means that for the output of the model, we discretize the 10s into 1375 time-intervals (each one of length <img src="https://latex.codecogs.com/gif.latex?10/1375&space;\approx&space;0.0072">s) and try to predict for each of these intervals whether someone recently finished saying "activate." 

Consider also the 10000 number above. This corresponds to discretizing the 10sec clip into 10/10000 = 0.001 second itervals. 0.001 seconds is also called 1 millisecond, or 1ms. So when we say we are discretizing according to 1ms intervals, it means we are using 10,000 steps. 

### Generating a single training example

Because speech data is hard to acquire and label, you will synthesize your training data using the audio clips of activates, negatives, and backgrounds. It is quite slow to record lots of 10 second audio clips with random "activates" in it. Instead, it is easier to record lots of positives and negative words, and record background noise separately (or download background noise from free online sources). 

To synthesize a single training example, you will:

- Pick a random 10 second background audio clip
- Randomly insert 0-4 audio clips of "activate" into this 10sec clip
- Randomly insert 0-2 audio clips of negative words into this 10sec clip

Because you had synthesized the word "activate" into the background clip, you know exactly when in the 10sec clip the "activate" makes its appearance. You'll see later that this makes it easier to generate the labels <img src="https://latex.codecogs.com/gif.latex?y^{\langle&space;t&space;\rangle}"> as well. 

You will use the pydub package to manipulate audio. Pydub converts raw audio files into lists of Pydub data structures (it is not important to know the details here). Pydub uses 1ms as the discretization interval (1ms is 1 millisecond = 1/1000 seconds) which is why a 10sec clip is always represented using 10,000 steps. 

**Overlaying positive/negative words on the background**:

Given a 10sec background clip and a short audio clip (positive or negative word), you need to be able to "add" or "insert" the word's short audio clip onto the background. To ensure audio segments inserted onto the background do not overlap, you will keep track of the times of previously inserted audio clips. You will be inserting multiple clips of positive/negative words onto the background, and you don't want to insert an "activate" or a random word somewhere that overlaps with another clip you had previously added. 

For clarity, when you insert a 1sec "activate" onto a 10sec clip of cafe noise, you end up with a 10sec clip that sounds like someone sayng "activate" in a cafe, with "activate" superimposed on the background cafe noise. You do *not* end up with an 11 sec clip. You'll see later how pydub allows you to do this. 

**Creating the labels at the same time you overlay**:

Recall also that the labels <img src="https://latex.codecogs.com/gif.latex?y^{\langle&space;t&space;\rangle}"> represent whether or not someone has just finished saying "activate." Given a background clip, we can initialize <img src="https://latex.codecogs.com/gif.latex?y^{\langle&space;t&space;\rangle}=0"> for all $t$, since the clip doesn't contain any "activates." 

When you insert or overlay an "activate" clip, you will also update labels for <img src="https://latex.codecogs.com/gif.latex?y^{\langle&space;t&space;\rangle}">, so that 50 steps of the output now have target label 1. You will train a GRU to detect when someone has *finished* saying "activate". For example, suppose the synthesized "activate" clip ends at the 5sec mark in the 10sec audio---exactly halfway into the clip. Recall that <img src="https://latex.codecogs.com/gif.latex?T_y&space;=&space;1375">, so timestep <img src="https://latex.codecogs.com/gif.latex?687&space;="> `int(1375*0.5)` corresponds to the moment at 5sec into the audio. So, you will set <img src="https://latex.codecogs.com/gif.latex?y^{\langle&space;688&space;\rangle}&space;=&space;1">. Further, you would quite satisfied if the GRU detects "activate" anywhere within a short time-internal after this moment, so we actually set 50 consecutive values of the label <img src="https://latex.codecogs.com/gif.latex?y^{\langle&space;t&space;\rangle}"> to 1. Specifically, we have <img src="https://latex.codecogs.com/gif.latex?y^{\langle&space;688&space;\rangle}&space;=&space;y^{\langle&space;689&space;\rangle}&space;=&space;\cdots&space;=&space;y^{\langle&space;737&space;\rangle}&space;=&space;1">.  

This is another reason for synthesizing the training data: It's relatively straightforward to generate these labels $<img src="https://latex.codecogs.com/gif.latex?y^{\langle&space;t&space;\rangle}"> as described above. In contrast, if you have 10sec of audio recorded on a microphone, it's quite time consuming for a person to listen to it and mark manually exactly when "activate" finished. 

Here's a figure illustrating the labels <img src="https://latex.codecogs.com/gif.latex?y^{\langle&space;t&space;\rangle}">, for a clip which we have inserted "activate", "innocent", activate", "baby." Note that the positive labels "1" are associated only with the positive words. 

<img src="images/label_diagram.png" style="width:500px;height:200px;">

To implement the training set synthesis process, you will use the following helper functions. All of these function will use a 1ms discretization interval, so the 10sec of audio is alwsys discretized into 10,000 steps. 

1. `get_random_time_segment(segment_ms)` gets a random time segment in our background audio
2. `is_overlapping(segment_time, existing_segments)` checks if a time segment overlaps with existing segments
3. `insert_audio_clip(background, audio_clip, existing_times)` inserts an audio segment at a random time in our background audio using `get_random_time_segment` and `is_overlapping`
4. `insert_ones(y, segment_end_ms)` inserts 1's into our label vector y after the word "activate"

Next, suppose you have inserted audio clips at segments (1000,1800) and (3400,4500). I.e., the first segment starts at step 1000, and ends at step 1800. Now, if we are considering inserting a new audio clip at (3000,3600) does this overlap with one of the previously inserted segments? In this case, (3000,3600) and (3400,4500) overlap, so we should decide against inserting a clip here. 

For the purpose of this function, define (100,200) and (200,250) to be overlapping, since they overlap at timestep 200. However, (100,199) and (200,250) are non-overlapping. 

Implement `is_overlapping(segment_time, existing_segments)` to check if a new time segment overlaps with any of the previous segments. You will need to carry out 2 steps:

1. Create a "False" flag, that you will later set to "True" if you find that there is an overlap.
2. Loop over the previous_segments' start and end times. Compare these times to the segment's start and end times. If there is an overlap, set the flag defined in (1) as True. You can use:
```python
for ....:
        if ... <= ... and ... >= ...:
            ...
```
There is overlap if the segment starts before the previous segment ends, and the segment ends after the previous segment starts.

Implement `insert_audio_clip()` to overlay an audio clip onto the background 10sec clip. You will need to carry out 4 steps:

1. Get a random time segment of the right duration in ms.
2. Make sure that the time segment does not overlap with any of the previous time segments. If it is overlapping, then go back to step 1 and pick a new time segment.
3. Add the new time segment to the list of existing time segments, so as to keep track of all the segments you've inserted.  
4. Overlay the audio clip over the background using pydub. We have implemented this for you.

Implement `insert_ones()`. You can use a for loop. (If you are an expert in python's slice operations, feel free also to use slicing to vectorize this.) If a segment ends at `segment_end_ms` (using a 10000 step discretization), to convert it to the indexing for the outputs <img src="https://latex.codecogs.com/gif.latex?y"> (using a <img src="https://latex.codecogs.com/gif.latex?1375"> step discretization), we will use this formula:  
```
    segment_end_y = int(segment_end_ms * Ty / 10000.0)
```
Implement `create_training_example()`. You will need to carry out the following steps:

1. Initialize the label vector <img src="https://latex.codecogs.com/gif.latex?y"> as a numpy array of zeros and shape <img src="https://latex.codecogs.com/gif.latex?(1,&space;T_y)">.
2. Initialize the set of existing segments to an empty list.
3. Randomly select 0 to 4 "activate" audio clips, and insert them onto the 10sec clip. Also insert labels at the correct position in the label vector <img src="https://latex.codecogs.com/gif.latex?y">.
4. Randomly select 0 to 2 negative audio clips, and insert them into the 10sec clip. 

## Model
### Build the model

Here is the architecture we will use.

<img src="images/model.png" style="width:600px;height:600px;">

One key step of this model is the 1D convolutional step (near the bottom of Figure 3). It inputs the 5511 step spectrogram, and outputs a 1375 step output, which is then further processed by multiple layers to get the final <img src="https://latex.codecogs.com/gif.latex?T_y&space;=&space;1375"> step output. This layer plays a role similar to the 2D convolutions you saw in Course 4, of extracting low-level features and then possibly generating an output of a smaller dimension. 

Computationally, the 1-D conv layer also helps speed up the model because now the GRU  has to process only 1375 timesteps rather than 5511 timesteps. The two GRU layers read the sequence of inputs from left to right, then ultimately uses a dense+sigmoid layer to make a prediction for <img src="https://latex.codecogs.com/gif.latex?y^{\langle&space;t&space;\rangle}">. Because <img src="https://latex.codecogs.com/gif.latex?y"> is binary valued (0 or 1), we use a sigmoid output at the last layer to estimate the chance of the output being 1, corresponding to the user having just said "activate."

Note that we use a uni-directional RNN rather than a bi-directional RNN. This is really important for trigger word detection, since we want to be able to detect the trigger word almost immediately after it is said. If we used a bi-directional RNN, we would have to wait for the whole 10sec of audio to be recorded before we could tell if "activate" was said in the first second of the audio clip.  

Implementing the model can be done in four steps:
    
**Step 1**: CONV layer. Use `Conv1D()` to implement this, with 196 filters, 
a filter size of 15 (`kernel_size=15`), and stride of 4. [[See documentation.](https://keras.io/layers/convolutional/#conv1d)]

**Step 2**: First GRU layer. To generate the GRU layer, use:
```
X = GRU(units = 128, return_sequences = True)(X)
```
Setting `return_sequences=True` ensures that all the GRU's hidden states are fed to the next layer. Remember to follow this with Dropout and BatchNorm layers. 

**Step 3**: Second GRU layer. This is similar to the previous GRU layer (remember to use `return_sequences=True`), but has an extra dropout layer. 

**Step 4**: Create a time-distributed dense layer as follows: 
```
X = TimeDistributed(Dense(1, activation = "sigmoid"))(X)
```
This creates a dense layer followed by a sigmoid, so that the parameters used for the dense layer are the same for every time step. [[See documentation](https://keras.io/layers/wrappers/).]

## Useful Functions

- **Python**
    - slice: `y[0, segment_end_y + 1 : segment_end_y + 51] = 1` 

- **Keras:**
    - [Conv1D](https://keras.io/layers/convolutional/#conv1d)
    - [Layer wrappers](https://keras.io/layers/wrappers/)

## Summary

- Data synthesis ia an effective way to create a large training set for speech problems, specifically trigger word detection.
- Using a spectrogram and optionally a 1D conv layer is a common pre-processing step prior to passing audio data to an RNN, GRU or LSTM.
- An end-to-end deep learning approach can be used to built a very effective trigger word detection system.
