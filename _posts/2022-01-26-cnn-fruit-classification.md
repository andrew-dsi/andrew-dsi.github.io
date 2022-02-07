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

With the pipeline, and architecture in place - we are now ready to train the baseline network!

In the below code we:

* Specify the number of epochs for training
* Set a location for the trained network to be saved (architecture & parameters)
* Set a *ModelCheckPoint* callback to save the best network at any point during training (based upon validation accuracy)
* Train the network and save the results to an object called *history*

```python

# training parameters
num_epochs = 50
model_filename = 'models/fruits_cnn_v01.h5'

# callbacks
save_best_model = ModelCheckpoint(filepath = model_filename,
                                  monitor = 'val_accuracy',
                                  mode = 'max',
                                  verbose = 1,
                                  save_best_only = True)

# train the network
history = model.fit(x = training_set,
                    validation_data = validation_set,
                    batch_size = batch_size,
                    epochs = num_epochs,
                    callbacks = [save_best_model])

```
<br>
The ModelCheckpoint callback that has been put in place means that we do not just save the *final* network at epoch 50, but instead we save the *best* network, in terms of validation set performance - from *any point* during training.  In other words, at the end of each of the 50 epochs, Keras will assess the performance on the validation set and if is has not seen any improvement in performance it will do nothing.  If it does see an improvement however, it will update the network file that is saved on our hard-drive.

<br>
#### Analysis Of Training Results

As we saved our training process to the *history* object, we can now analyse the performance (Classification Accuracy, and Loss) of the network epoch by epoch.

```python

import matplotlib.pyplot as plt

# plot validation results
fig, ax = plt.subplots(2, 1, figsize=(15,15))
ax[0].set_title('Loss')
ax[0].plot(history.epoch, history.history["loss"], label="Training Loss")
ax[0].plot(history.epoch, history.history["val_loss"], label="Validation Loss")
ax[1].set_title('Accuracy')
ax[1].plot(history.epoch, history.history["accuracy"], label="Training Accuracy")
ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation Accuracy")
ax[0].legend()
ax[1].legend()
plt.show()

# get best epoch performance for validation accuracy
max(history.history['val_accuracy'])

```
<br>
The below image contains two plots, the first showing the epoch by epoch **Loss** for both the training set (blue) and the validation set (orange) & the second show the epoch by epoch **Classification Accuracy** again, for both the training set (blue) and the validation set (orange).

<br>
![alt text](/img/posts/cnn-baseline-accuracy-plot.png "CNN Baseline Accuracy Plot")

<br>
There are two key learnings from above plots. The first is that, with this baseline architecture & the parameters we set for training, we are reaching our best performance in around 10-20 epochs - after that, not much improvement is seen.  This isn't to say that 50 epochs is wrong, especially if we change our network - but is interesting to note at this point.

The second thing to notice is *very important* and that is the significant gap between orange and blue lines on the plot, in other words between our validation performance and our training performance.

This gap is over-fitting.

Focusing on the lower plot above (Classification Accuracy) - it appears that our network is learning the features of the training data *so well* that after about 20 or so epochs it is *perfectly* predicting those images - but on the validation set, it never passes approximately **83% Classification Accuracy**.

We do not want over-fitting! It means that we're risking our predictive performance on new data.  The network is not learning to generalise, meaning that if something slightly 
different comes along then it's going to really, really struggle to predict well, or at least predict reliably!

We will look to address this with some clever concepts, and you will see those in the next sections.

<br>
#### Performance On The Test Set

Above, we assessed our models performance on both the training set and the validation set - both of which were being passed in during training.

Here, we will get a view of how well our network performs when predict on data that was *no part* of the training process whatsoever - our test set.

A test set can be extremely useful when looking to assess many different iterations of a network we build.  Where the validation set might be sent through the model in slightly different orders during training in order to assess the epoch by epoch performance, our test set is a *static set* of images.  Because of this, it makes for a really good baseline for testing the first iteration of our network versus any subsequent versions that we create, perhaps after we refine the architecture, or add any other clever bits of
logic that we think might help the network perform better in the real world.

In the below code we run this in isolation from training.  We:

* Import the required packages for importing & manipulating our test set images
* Set up the parameters for the predictions
* Load in the saved network file from training
* Create a function for preprocessing our test set images in the same way that training & validation images were
* Create a function for making predictions, returning both predicted class label, and predicted class probability
* Iterate through our test set images, preprocessing each and passing to the network for prediction
* Create a Pandas DataFrame to hold all prediction data

```python

# import required packages
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd
from os import listdir

# parameters for prediction
model_filename = 'models/fruits_cnn_v01.h5'
img_width = 128
img_height = 128
labels_list = ['apple', 'avocado', 'banana', 'kiwi', 'lemon', 'orange']

# load model
model = load_model(model_filename)

# image pre-processing function
def preprocess_image(filepath):
    
    image = load_img(filepath, target_size = (img_width, img_height))
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)
    image = image * (1./255)
    
    return image

# image prediction function
def make_prediction(image):
    
    class_probs = model.predict(image)
    predicted_class = np.argmax(class_probs)
    predicted_label = labels_list[predicted_class]
    predicted_prob = class_probs[0][predicted_class]
    
    return predicted_label, predicted_prob

# loop through test data
source_dir = 'data/test/'
folder_names = ['apple', 'avocado', 'banana', 'kiwi', 'lemon', 'orange']
actual_labels = []
predicted_labels = []
predicted_probabilities = []
filenames = []

for folder in folder_names:
    
    images = listdir(source_dir + '/' + folder)
    
    for image in images:
        
        processed_image = preprocess_image(source_dir + '/' + folder + '/' + image)
        predicted_label, predicted_probability = make_prediction(processed_image)
        
        actual_labels.append(folder)
        predicted_labels.append(predicted_label)
        predicted_probabilities.append(predicted_probability)
        filenames.append(image)
        
# create dataframe to analyse
predictions_df = pd.DataFrame({"actual_label" : actual_labels,
                               "predicted_label" : predicted_labels,
                               "predicted_probability" : predicted_probabilities,
                               "filename" : filenames})

predictions_df['correct'] = np.where(predictions_df['actual_label'] == predictions_df['predicted_label'], 1, 0)

```
<br>
After running the code above, we end up with a Pandas DataFrame containing prediction data for each test set image. A random sample of this can be seen in the table below:

<br>
| **actual_label** | **predicted_label** | **predicted_probability** | **filename** | **correct** |
|---|---|---|---|---|
| apple | lemon | 0.700764 | apple_0034.jpg | 0 |
| avocado | avocado | 0.99292046 | avocado_0074.jpg | 1 |
| orange | orange | 0.94840413 | orange_0004.jpg | 1 |
| banana | lemon | 0.87131584 | banana_0024.jpg | 0 |
| kiwi | kiwi | 0.66800004 | kiwi_0094.jpg | 1 |
| lemon | lemon | 0.8490372 | lemon_0084.jpg | 1 |

<br>
In our data we have:

* Actual Label: The true label for that image
* Prediction Label: The predicted label for the image (from the network)
* Predicted Probability: The network's perceived probability for the predicted label
* Filename: The test set image on our local drive (for reference)
* Correct: A flag showing whether the predicted label is the same as the actual label

This dataset is extremely useful as we can not only calculate our classification accuracy, but we can also deep-dive into images where the network was struggling to predict and try to assess why - leading to us improving our network, and potentially our input data!

<br>
#### Test Set Classification Accuracy

Using our DataFrame, we can calculate our overall Test Set classification accuracy using the below code:

```python

# overall test set accuracy
test_set_accuracy = predictions_df['correct'].sum() / len(predictions_df)
print(test_set_accuracy)

```
<br>
Our baseline network acheives a **75% Classification Accuracy** on the Test Set.  It will be interesting to see how much improvement we can this with additions & refinements to our network.

<br>
#### Test Set Confusion Matrix

Overall Classification Accuracy is very useful, but it can hide what is really going on with the network's predictions!

As we saw above, our Classification Accuracy for the whole test set was 75%, but it might be that our network is predicting extremely well on apples, but struggling with Lemons as for some reason it is regularly confusing them with Oranges.  A Confusion Matrix can help us uncover insights like this!

We can create a Confusion Matrix with the below code:

```python

# confusion matrix (percentages)
confusion_matrix = pd.crosstab(predictions_df['predicted_label'], predictions_df['actual_label'], normalize = 'columns')
print(confusion_matrix)

```
<br>
This results in the following output:

```

actual_label     apple  avocado  banana  kiwi  lemon  orange
predicted_label                                             
apple              0.8      0.0     0.0   0.1    0.0     0.1
avocado            0.0      1.0     0.0   0.0    0.0     0.0
banana             0.0      0.0     0.2   0.1    0.0     0.0
kiwi               0.0      0.0     0.1   0.7    0.0     0.0
lemon              0.2      0.0     0.7   0.0    1.0     0.1
orange             0.0      0.0     0.0   0.1    0.0     0.8

```
<br>
Along the top are our *actual* classes and down the side are our *predicted* classes - so counting *down* the columns we can get the Classification Accuracy (%) for each class, and we can see where it is getting confused.

So, while overall our test set accuracy was 75% - for each individual class we see:

* Apple: 80%
* Avocado: 100%
* Banana: 20%
* Kiwi: 70%
* Lemon: 100%
* Orange: 80%

This is very powerful - we now can see what exactly is driving our *overall* Classification Accuracy.

The standout insight here is for Bananas - with a 20% Classification Accuracy, and even more interestingly we can see where it is getting confused. The network predicted 70% of Banana images to be of the class Lemon!

___
<br>
# Tackling Overfitting With Dropout <a name="cnn-dropout"></a>

<br>
#### Dropout Overview

Dropout is a technique used in Deep Learning primarily to reduce the effects of over-fitting. Over-fitting is where the network learns the patterns of the training data so specifically, that it runs the risk of not generalising well, and being very unreliable when used to predict on new, unseen data.

Dropout works in a way where, for each batch of observations that is sent forwards through the network, a pre-specified proportion of the neurons in a hidden layer are essentially ignored or deactivated.  This can be applied to any number of the hidden layers.

When we apply Dropout, the deactivated neurons are completely taken out of the picture - they take no part in the passing of information through the network.

All the math is the same, the network will process everything as it always would (so taking the sum of the inputs multiplied by the weights, and adding a bias term, applying activation functions, and updating the network’s parameters using Back Propagation) - it’s just that in this scenario where we are disregarding some of the neurons, we’re essentially pretending that they’re not there.

In a full network (i.e. where Dropout is not being applied) each of the combinations of neurons becomes quite specific at what it represents, at least in terms of predicting the output.  At a high level, if we were classifying pictures of cats and dogs, there might be some linked combination of neurons that fires when it sees pointy ears and a long tongue.  This combination of neurons becomes very tuned into its role in prediction, and it becomes very good at what it does - but as is the definition of overfitting, it becomes too good - it becomes too rigidly aligned with the training data.

If we *drop out* neurons during training, *other* neurons need to jump in fill in for this particular role of detecting those features.  They essentially have to come in at late notice and cover the ignored neurons job, dealing with that particular representation that is so useful for prediction.

Over time, with different combinations of neurons being ignored for each mini-batch of data - the network becomes more adept at generalising and thus is less likely to overfit to the training data.  Since no particular neuron can rely on the presence of other neurons, and the features with which they represent - the network learns more robust features, and are less susceptible to noise.

In a Convolutional Neural Network, such as in our task here - it is generally best practice to only apply Dropout to the Dense (Fully Connected) layer or layers, rather than to the Convolutional Layers.  


<br>
#### Updated Network Architecture

In our task here, we only have one Dense Layer, so we apply Dropout to that layer only.  A common proportion to apply (i.e. the proportion of neurons in the layer to be deactivated randomly each pass) is 0.5 or 50%.  We will apply this here.

```python

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
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

```

<br>
#### Training The Updated Network

We run the exact same code to train this updated network as we did for the baseline network (50 epochs) - the only change is that we modify the filename for the saved network to ensure we have all network files for comparison.

<br>
#### Analysis Of Training Results

As we again saved our training process to the *history* object, we can now analyse & plot the performance (Classification Accuracy, and Loss) of the updated network epoch by epoch.

With the baseline network we saw very strong overfitting in action - it will be interesting to see if the addition of Dropout has helped!

The below image shows the same two plots we analysed for the updated network, the first showing the epoch by epoch **Loss** for both the training set (blue) and the validation set (orange) & the second show the epoch by epoch **Classification Accuracy** again, for both the training set (blue) and the validation set (orange).

<br>
![alt text](/img/posts/cnn-dropout-accuracy-plot.png "CNN Dropout Accuracy Plot")

<br>
Firstly, we can see a peak Classification Accuracy on the validation set of around **89%** which is higher than the **83%** we saw for the baseline network.

Secondly, and what we were really looking to see, is that gap between the Classification Accuracy on the training set, and the validation set has been mostly eliminated. The two lines are trending up at more or less the same rate across all epochs of training - and the accuracy on the training set also never reach 100% as it did before meaning that we are indeed seeing this *generalisation* that we want!

The addition of Dropout does appear to have remedied the overfitting that we saw in the baseline network.  This is because, while some neurons are turned off during each mini-batch iteration of training - all will have their turn, many times, to be updated - just in a way where no neuron, or combination of neurons will become so hard-wired to certain features found in the training data!

<br>
#### Performance On The Test Set

During training, we assessed our updated networks performance on both the training set and the validation set.  Here, like we did for the baseline network, we will get a view of how well our network performs when predict on data that was *no part* of the training process whatsoever - our test set.

We run the exact same code as we did for the baseline network, with the only change being to ensure we are loading in network file for the updated network

<br>
#### Test Set Classification Accuracy

Using our DataFrame, we can calculate our overall Test Set classification accuracy using the below code:

```python

# overall test set accuracy
test_set_accuracy = predictions_df['correct'].sum() / len(predictions_df)
print(test_set_accuracy)

```
<br>
Our baseline network acheives a **75% Classification Accuracy** on the Test Set.  It will be interesting to see how much improvement we can this with additions & refinements to our network.

<br>
#### Test Set Confusion Matrix

Overall Classification Accuracy is very useful, but it can hide what is really going on with the network's predictions!

As we saw above, our Classification Accuracy for the whole test set was 75%, but it might be that our network is predicting extremely well on apples, but struggling with Lemons as for some reason it is regularly confusing them with Oranges.  A Confusion Matrix can help us uncover insights like this!

We can create a Confusion Matrix with the below code:

```python

# confusion matrix (percentages)
confusion_matrix = pd.crosstab(predictions_df['predicted_label'], predictions_df['actual_label'], normalize = 'columns')
print(confusion_matrix)

```
<br>
This results in the following output:

```

actual_label     apple  avocado  banana  kiwi  lemon  orange
predicted_label                                             
apple              0.8      0.0     0.0   0.1    0.0     0.1
avocado            0.0      1.0     0.0   0.0    0.0     0.0
banana             0.0      0.0     0.2   0.1    0.0     0.0
kiwi               0.0      0.0     0.1   0.7    0.0     0.0
lemon              0.2      0.0     0.7   0.0    1.0     0.1
orange             0.0      0.0     0.0   0.1    0.0     0.8

```
<br>
Along the top are our *actual* classes and down the side are our *predicted* classes - so counting *down* the columns we can get the Classification Accuracy (%) for each class, and we can see where it is getting confused.

So, while overall our test set accuracy was 75% - for each individual class we see:

* Apple: 80%
* Avocado: 100%
* Banana: 20%
* Kiwi: 70%
* Lemon: 100%
* Orange: 80%

This is very powerful - we now can see what exactly is driving our *overall* Classification Accuracy.

The standout insight here is for Bananas - with a 20% Classification Accuracy, and even more interestingly we can see where it is getting confused. The network predicted 70% of Banana images to be of the class Lemon!


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
