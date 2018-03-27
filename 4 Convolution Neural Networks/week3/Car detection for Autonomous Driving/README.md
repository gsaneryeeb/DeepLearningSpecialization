# Autonomous driving - Car detection

- Use object detection on a car detection dataset
- Deal with bounding boxes

## YOLO

### Model details

- The **input** is a batch of images of shape(m,608,608,3)
- The **output** is a list of bounding boxes along with the recognized classes. Each bounding box is represented by 6 numbers $(p_c, b_x, b_y, b_h, b_w, c)$ as explained above. If you expand $c$ into an 80-dimensional vector, each bounding box is then represented by 85 numbers. 

We will use 5 anchor boxes. So you can think of the YOLO architecture as the following: IMAGE (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85).

- Non-max suppression
    - Get rid of boxes with a low score(meaning, the box is not very cofident about detecting a class)
    - Select only one box when several boxes overlap with each other and detect the same object.

- Compute box scores by doing the elementwise product.
- For each box, find:
    - the index of the class with the maximum box score. Using [argmax](https://keras.io/backend/#argmax) (Be careful with what axis you choose; consider using axis=-1)
    - the corresponding box score. Using [max](https://keras.io/backend/#max) (Be careful with what axis you choose; consider using axis=-1)
    - Create a mask by using a threshold. As a reminder: `([0.9, 0.3, 0.4, 0.5, 0.1] < 0.4)` returns: `[False, True, False, False, True]`. The mask should be True for the boxes you want to keep. 
    - Use TensorFlow to apply the mask to box_class_scores, boxes and box_classes to filter out the boxes we don't want. You should be left with just the subset of boxes you want to keep. ([boolean_mask](https://www.tensorflow.org/api_docs/python/tf/boolean_mask))
- **Intersection over Union**(IoU)
<img src="nb_images/iou.png" style="width:500px;height:400;">

### Funtions

- [tf.image.non_max_suppression()](https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression)
- [K.gather()](https://www.tensorflow.org/api_docs/python/tf/gather)

### Summary for YOLO
- Input image (608, 608, 3)
- The input image goes through a CNN, resulting in a (19,19,5,85) dimensional output.
- After flattening the output is a volume of shape (19,19,425):
    - Each cell in a 19x19 grid over the input image gives 425 numbers.
    - 425 = 5 x 85 because each cell contains predictions for 5 boxes, corresponding to 5 anchor boxes, as seen in lecture.
    - 85 = 5 + 80 where 5 is because ( <img src="https://latex.codecogs.com/gif.latex?_{p_{c}}">, <img src="https://latex.codecogs.com/gif.latex?_{b_{x}}">,<img src="https://latex.codecogs.com/gif.latex?_{b_{y}}">,<img src="https://latex.codecogs.com/gif.latex?_{b_{h}}">,<img src="https://latex.codecogs.com/gif.latex?_{b_{w}}">) has 5 numbers, and 80 is the number of classes we'd like to detect
- You then select only few boxes based on:
    - Score-thresholding: throw away boxes that have detected a class with a score less than the threshold
    - Non-max supression: Compute the Intersection over Union and avoid selectiong overlapping boxes
- This gives you YOLO's final output.

### Summary

- YOLO is a state-of-the-art object detection model that is fast and accurate
- It runs an input image through a CNN which outputs a 19x19x5x85 dimensional volume.
- The encoding can be seen as a grid where each of the 19x19 cells contains information about 5 boxes.
- You filter through all the boxes using non-max suppressing. Specifically:
    - Score thresholding on the probability of detecting a class to keep only accurate(high probability) boxes
    - Intersection over Union (IoU) thresholding to eliminate overlapping boxes
- Because training a YOLO model from randomly initialized weights is non-trivial and requires a large dataset as well as lot of computation, we used previously trained model parameters in this exercise.