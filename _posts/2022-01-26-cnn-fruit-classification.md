---
layout: post
title: Fruit Classification Using A Convolutional Neural Network
image: "/posts/cnn-fruit-classification-title-img.png"
tags: [Deep Learning, CNN, Data Science, Computer Vision, Python]
---

In this project we build & optimise a Convolutional Neural Network to classify images of fruits, with the goal of helping a grocery retailer enhance & scale their sorting & delivery processes. 

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [01. Data Overview](#data-overview)
- [02. Data Pipeline](#data-pipeline)
- [03. CNN Overview](#cnn-overview)
- [04. Baseline Network](#cnn-baseline)
- [05. Tackling Overfitting With Dropout](#cnn-dropout)
- [06. Image Augmentation](#cnn-augmentation)
- [07. Hyper-Parameter Tuning](#cnn-tuning)
- [08. Transfer Learning](#cnn-transfer-learning)
- [09. Overall Results Discussion](#cnn-results)
- [10. Next Steps & Growth](#growth-next-steps)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Our client had an interesting proposal put forward to them, and requested our help to assess whether it was viable.

At a recent tech conference, they spoke to a contact from a robotics company that creates robotic solutions that help other businesses scale and optimise their operations.

Their representative mentioned that they had built a prototype for a robotic sorting arm that could be used to pick up and move products off a platform.  It would use a camera to "see" the product, and could be programmed to move that particular product into a designated bin, for further processing.

The only thing they hadn't figured out was how to actually identify each product using the camera, so that the robotic arm could move it to the right place.

We were asked to put forward a proof of concept on this - and were given some sample images of fruits from their processing platform.

If this was successful and put into place on a larger scale, the client would be able to enhance their sorting & delivery processes.

<br>
<br>
### Actions <a name="overview-actions"></a>

We utilise the *Keras* Deep Learning library for this task.

We start by creating our pipeline for feeding training & validation images in batches, from our local directory, into the network.

Our baseline network is simple, but gives us a starting point to refine from.  This network contains **2 Convolutional Layers**, each with **32 filters** and subsequent **Max Pooling** Layers.  We have a **single Dense (Fully Connected) layer** following flattening with **32 neurons** followed by our output layer.  We apply the **relu** activation function on all layers, and use the **adam** optimizer.

Our first refinement is to add **Dropout** to tackle the issue of overfitting which is prevalent in the baseline network performance.  We use a **dropout rate of 0.5**.

We then add in **Image Augmentation** to our data pipeline to increase the variation of input images for the network to learn from, resulting in a more robust results as well as also address overfitting.

With these additions in place, we utlise *keras-tuner* to optimise our network architecture & tune the hyperparameters.  The best network from this testing contains **3 Convolutional Layers**, each followed by **Max Pooling** Layers.  The first Convolutional Layer has **96 filters**, the second & third have **64 filters**.  The output of this third layer is flattened and passed to a **single Dense (Fully Connected) layer** with **160 neurons**.  The Dense Layer has **Dropout** applied with a **dropout rate of 0.5**.  The output from this is passed to the output layer.  Again, we apply the **relu** activation function on all layers, and use the **adam** optimizer.

Finally, we utilise **Transfer Learning** to compare our network's results against that of the pre-trained **VGG16** network.

<br>
<br>

### Results <a name="overview-results"></a>

xxx
xxx
xxx
xxx
xxx

<br>
<br>
### Growth/Next Steps <a name="overview-growth"></a>

Next Steps:  Showcase to client, discuss what made the network more robust, get more data/classes

Growth: Try other networks for transfer learning, more epochs, different batch sizes etc

<br>
<br>

___

# Data Overview  <a name="data-overview"></a>

To build out this proof of concept, the client have provided us some sample data. This is made up of images of six different types of fruit, sitting on the landing platform in the warehouse.

We randomly split the images for each fruit into training (60%), validation (30%) and test (10%) sets.

Examples of four images of each fruit class can be seen in the image below:

<br>
![alt text](/img/posts/cnn-image-examples.png "CNN Fruit Classification Samples")

<br>
For ease of use in Keras, our folder structure first splits into training, validation, and test directories, and within each of those is split again into directories based upon the six fruit classes.

All images are of size 300 x 200 pixels.

___
<br>
# Data Pipeline  <a name="data-pipeline"></a>

Before we get to building the network architecture, & subsequently training & testing it - we need to set up a pipeline for our images to flow through, from our local hard-drive where they are located, to, and through our network.

In the code below, we:

* Import the required packages
* Set up the parameters for our pipeline
* Set up our image generators to process the images as they come in
* Set up our generator flow - specifying what we want to pass in for each iteration of training

<br>
```python

# import the required python libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# data flow parameters
training_data_dir = 'data/training'
validation_data_dir = 'data/validation'
batch_size = 32
img_width = 128
img_height = 128
num_channels = 3
num_classes = 6

# image generators
training_generator = ImageDataGenerator(rescale = 1./255)
validation_generator = ImageDataGenerator(rescale = 1./255)

# image flows
training_set = training_generator.flow_from_directory(directory = training_data_dir,
                                                      target_size = (img_width, img_height),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical')

validation_set = validation_generator.flow_from_directory(directory = validation_data_dir,
                                                                      target_size = (img_width, img_height),
                                                                      batch_size = batch_size,
                                                                      class_mode = 'categorical')

```
<br>
We specify that we will resize the images down to 128 x 128 pixels, and that we will pass in 32 images at a time (known as the batch size) for training.

To start with, we simply use the generators to rescale the raw pixel values (ranging between 0 and 255) to float values that exist between 0 and 1.  The reason we do this is mainly to help Gradient Descent find an optimal, or near optional solution each time much more efficiently - in other words, it means that the features that are learned in the depths of the network are of a similar magnitude, and the learning rate that is applied to descend down the loss or cost function across many dimensions, is somewhat proportionally similar across all dimensions - and long story short, means training time is faster as Gradient Descent can converge faster each time!

We will add more logic to the training set generator to apply Image Augmentation.

With this pipeline in place, our images will be extracted, in batches of 32, from our hard-drive, where they're being stored and sent into our model for training!

___
<br>
# Convolutional Neural Network Overview <a name="cnn-overview"></a>

Convolutional Neural Networks (CNN) are an adaptation of Artificial Neural Networks and are primarily used for image based tasks.

To a computer, an image is simply made up of numbers, those being the colour intensity values for each pixel.  Colour images have values ranging between 0 and 255 for each pixel, but have three of these values, for each - one for Red, one for Green, and one for Blue, or in other words the RGB values that mix together to make up the specific colour of each pixel.

These pixel values are the *input* for a Convolutional Neural Network.  It needs to make sense of these values to make predictions about the image, for example, in our task here, to predict what the image is of, one of the six possible fruit classes.

The pixel values themselves don't hold much useful information on their own - so the network needs to turn them into *features* much like we do as humans.

A big part of this process is called **Convolution** where each input image is scanned over, and compared to many different, and smaller filters, to compress the image down into something more generalised.  This process not only helps reduce the problem space, it also helps reduce the network's sensitivy to minor changes, in other words to know that two images are of the same object, even though the images are not *exactly* the same.

A somewhat similar process called **Pooling** is also applied to faciliate this *generalisation* even further.  A CNN can contain many of these Convolution & Pooling layers - with deeper layers finding more abstract features.

Similar to Artificial Neural Networks, Activation Functions are applied to the data as it moves forward through the network, helping the network decide which neurons will fire, or in other words, helping the network understand which neurons are more or less important for different features, and ultimately which neurons are more or less important for the different output classes.

Over time - as a Convolutional Neural Network trains, it iteratively calculates how well it is predicting on the known classes we pass it (known as the **loss** or **cost**, then heads back through in a process known as **Back Propagation** to update the paramaters within the network, in a way that reduces the error, or in other words, improves the match between predicted outputs and actual outputs.  Over time, it learns to find a good mapping between the input data, and the output classes.

There are many parameters that can be changed within the architecture of a Convolutional Neural Network, as well as clever logic that can be included, all which can affect the predictive accuracy.  We will discuss and put in place many of these below!

___
<br>
# Baseline Network <a name="cnn-baseline"></a>

<br>
#### Network Architecture

Our baseline network is simple, but gives us a starting point to refine from.  This network contains **2 Convolutional Layers**, each with **32 filters** and subsequent **Max Pooling** Layers.  We have a **single Dense (Fully Connected) layer** following flattening with **32 neurons** followed by our output layer.  We apply the **relu** activation function on all layers, and use the **adam** optimizer.

```python

# network architecture
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same', input_shape = (img_width, img_height, num_channels)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(32))
model.add(Activation('relu'))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

# compile network
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# view network architecture
model.summary()

```
<br>
The below shows us more clearly our baseline architecture:

```

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 128, 128, 32)      896       
_________________________________________________________________
activation (Activation)      (None, 128, 128, 32)      0         
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 64, 64, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 64, 64, 32)        9248      
_________________________________________________________________
activation_1 (Activation)    (None, 64, 64, 32)        0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 32, 32, 32)        0         
_________________________________________________________________
flatten (Flatten)            (None, 32768)             0         
_________________________________________________________________
dense (Dense)                (None, 32)                1048608   
_________________________________________________________________
activation_2 (Activation)    (None, 32)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 6)                 198       
_________________________________________________________________
activation_3 (Activation)    (None, 6)                 0         
=================================================================
Total params: 1,058,950
Trainable params: 1,058,950
Non-trainable params: 0
_________________________________________________________________


```

<br>
#### Training The Network

xxx

```python

# xxx
xxx

```
<br>
xxx

<br>
#### Analysis Of Training Results

xxx

```python

# xxx
xxx

```
<br>
xxx

<br>
#### Performance On The Test Set

xxx

```python

# xxx
xxx

```
<br>
xxx

___
<br>
# Tackling Overfitting With Dropout <a name="cnn-dropout"></a>

<br>
#### Dropout Overview

xxx
xxx

<br>
#### Network Architecture

xxx

```python

# xxx
xxx

```
<br>
xxx

<br>
#### Training The Network

xxx

```python

# xxx
xxx

```
<br>
xxx

<br>
#### Analysis Of Training Results

xxx

```python

# xxx
xxx

```
<br>
xxx

<br>
#### Performance On The Test Set

xxx

```python

# xxx
xxx

```
<br>
xxx


___
<br>
# Image Augmentation <a name="cnn-augmentation)"></a>

<br>
#### Image Augmentation Overview

xxx
xxx

<br>
#### Network Architecture

xxx

```python

# xxx
xxx

```
<br>
xxx

<br>
#### Training The Network

xxx

```python

# xxx
xxx

```
<br>
xxx

<br>
#### Analysis Of Training Results

xxx

```python

# xxx
xxx

```
<br>
xxx

<br>
#### Performance On The Test Set

xxx

```python

# xxx
xxx

```
<br>
xxx

___
<br>
# Hyper-Parameter Tuning <a name="cnn-tuning))"></a>

<br>
#### Keras Tuner Overview

xxx
xxx

<br>
#### Network Architecture

xxx

```python

# xxx
xxx

```
<br>
xxx

<br>
#### Training The Network

xxx

```python

# xxx
xxx

```
<br>
xxx

<br>
#### Analysis Of Training Results

xxx

```python

# xxx
xxx

```
<br>
xxx

<br>
#### Performance On The Test Set

xxx

```python

# xxx
xxx

```
<br>
xxx


___
<br>
# Transfer Learning With VGG16 <a name="cnn-transfer-learning))"></a>

<br>
#### Transfer Learning Overview

xxx
xxx

<br>
#### Network Architecture

xxx

```python

# xxx
xxx

```
<br>
xxx

<br>
#### Training The Network

xxx

```python

# xxx
xxx

```
<br>
xxx

<br>
#### Analysis Of Training Results

xxx

```python

# xxx
xxx

```
<br>
xxx

<br>
#### Performance On The Test Set

xxx

```python

# xxx
xxx

```
<br>
xxx

___
<br>
# Overall Results Discussion <a name="cnn-results"></a>

xxx
xxx

___
<br>
# Growth & Next Steps <a name="growth-next-steps"></a>

xxx
xxx
