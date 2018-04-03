# Operations on word vectors
**Goal**
- Load pre-trained word vectors, and measure similarity using cosine similarity
- Use word embeddings to solve word analogy problems such as Man is to Woman as King is to __.
- Modify word embeddings to reduce their gender bias

## Cosine similarity
Given two vectors <img src="https://latex.codecogs.com/gif.latex?u"> and <img src="https://latex.codecogs.com/gif.latex?v">, cosine similarity is defined as follows:

<img src="https://latex.codecogs.com/gif.latex?CosineSimilarity(u,v)=\frac&space;{u&space;.&space;v}{||u||_2&space;||v||_2}&space;=&space;cos(\theta&space;)">

where <img src="https://latex.codecogs.com/gif.latex?u.v"> is the dot product (or inner product) of two vectors, <img src="https://latex.codecogs.com/gif.latex?||u||_2"> is the norm (or length) of the vector <img src="https://latex.codecogs.com/gif.latex?u">, and <img src="https://latex.codecogs.com/gif.latex?\theta"> is the angle between <img src="https://latex.codecogs.com/gif.latex?u"> and <img src="https://latex.codecogs.com/gif.latex?v">. This similarity depends on the angle between <img src="https://latex.codecogs.com/gif.latex?u"> and <img src="https://latex.codecogs.com/gif.latex?v">. If <img src="https://latex.codecogs.com/gif.latex?u"> and <img src="https://latex.codecogs.com/gif.latex?v"> are very similar, their cosine similarity will be close to 1; if they are dissimilar, the cosine similarity will take a smaller value. If the two vectors are similar but opposite, the <img src="https://latex.codecogs.com/gif.latex?\theta"> is close to <img src="https://latex.codecogs.com/gif.latex?180^{\circ}"> and <img src="https://latex.codecogs.com/gif.latex?cos(\theta&space;)&space;\approx&space;-1">

**Reminder:** The norm of <img src="https://latex.codecogs.com/gif.latex?u"> is defined as <img src="https://latex.codecogs.com/gif.latex?||u||_2&space;=&space;\sqrt{\sum_{i=1}^{n}&space;u_i^2}">

## Word analogy task
In the word analogy task, we complete the sentence <font color='brown'>"*a* is to *b* as *c* is to **____**"</font>. An example is <font color='brown'> '*man* is to *woman* as *king* is to *queen*' </font>. In detail, we are trying to find a word *d*, such that the associated word vectors <img src="https://latex.codecogs.com/gif.latex?e_a,&space;e_b,&space;e_c,&space;e_d"> are related in the following manner: <img src="https://latex.codecogs.com/gif.latex?e_b&space;-&space;e_a&space;\approx&space;e_d&space;-&space;e_c">. We will measure the similarity between <img src="https://latex.codecogs.com/gif.latex?e_b&space;-&space;e_a"> and <img src="https://latex.codecogs.com/gif.latex?e_d&space;-&space;e_c"> using cosine similarity. 

## Debiasing word vectors

### Neturalize bias for non-gender specific words

<img src = "https://latex.codecogs.com/gif.latex?e^{debiased}&space;=&space;\frac{e\cdot&space;g}{^{\left&space;\|&space;g&space;\right&space;\|_{2}^{2}}}&space;\ast&space;g">

<img src="https://latex.codecogs.com/gif.latex?e^{debiased}&space;=&space;e&space;-&space;e^{bias\_component}">

### Equalization algorithm for gender-specific words

The key equations are:

<img src="https://latex.codecogs.com/gif.latex?\mu&space;=&space;\frac{e_{w1}&space;&plus;&space;e_{w2}}{2}\">

<img src="https://latex.codecogs.com/gif.latex?_{\mu&space;_{B}}=&space;\frac{\mu&space;\cdot&space;bias\_axis}{_{\left&space;\|&space;bias\_axis&space;\right&space;\|_{2}^{2}}}&space;\ast&space;bias\_axis">

<img src="https://latex.codecogs.com/gif.latex?\mu_{\perp}&space;=&space;\mu&space;-&space;\mu_{B}">

<img src="https://latex.codecogs.com/gif.latex?_{e_{w1B}}&space;=&space;\frac{_{e_{w1}}\cdot&space;bias\_axis}{_{\left&space;\|&space;bias\_axis&space;\right&space;\|_{2}^{2}}}&space;\ast&space;bias\_axis">

<img src="https://latex.codecogs.com/gif.latex?_{e_{w2B}}&space;=&space;\frac{_{e_{w2}}\cdot&space;bias\_axis}{_{\left&space;\|&space;bias\_axis&space;\right&space;\|_{2}^{2}}}&space;\ast&space;bias\_axis">

<img src="https://latex.codecogs.com/gif.latex?e_{w1B}^{corrected}&space;=&space;\sqrt{&space;|{1&space;-&space;||\mu_{\perp}&space;||^2_2}&space;|}&space;*&space;\frac{e_{\text{w1B}}&space;-&space;\mu_B}&space;{|(e_{w1}&space;-&space;\mu_{\perp})&space;-&space;\mu_B)|}">

<img src="https://latex.codecogs.com/gif.latex?e_{w2B}^{corrected}&space;=&space;\sqrt{&space;|{1&space;-&space;||\mu_{\perp}&space;||^2_2}&space;|}&space;*&space;\frac{e_{\text{w2B}}&space;-&space;\mu_B}&space;{|(e_{w2}&space;-&space;\mu_{\perp})&space;-&space;\mu_B)|}">

<img src="https://latex.codecogs.com/gif.latex?e_1&space;=&space;e_{w1B}^{corrected}&space;&plus;&space;\mu_{\perp}">

<img src="https://latex.codecogs.com/gif.latex?e_2&space;=&space;e_{w2B}^{corrected}&space;&plus;&space;\mu_{\perp}">



## Summary
- Cosine similarity a good way to compare similarity between pairs of word vectors.(Though L2 disantace works too.)
- For NLP applications, using a pre-trained set of word vectors from the internet is often a good way to get started.

