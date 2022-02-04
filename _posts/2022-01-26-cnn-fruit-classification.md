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
- [07. Transfer Learning](#cnn-transfer-learning)
- [08. Hyper-Parameter Tuning](#cnn-tuning)
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

Convolutional Neural Networks are an adaptation of Artificial Neural Networks and are primarily used for image based tasks.

To a computer, an image is simply made up of numbers, those being the colour intensity values for each pixel.  Colour images have values ranging between 0 and 255 for each pixel, but have three of these values, for each - one for Red, one for Green, and one for Blue, or in other words the RGB values that mix together to make up the specific colour of each pixel.

These pixel values are the *input* for a Convolutional Neural Network.  It needs to make sense of these values to make predictions about the image, for example, in our task here, to predict what the image is of, one of the six possible fruit classes.

The pixel values themselves don't hold much useful information on their own - so the network needs to turn them into *features* much like we do as humans.

A big part of this process is called **Convolution** where each input image is scanned over, and compared to many different, and smaller filters, to compress the image down into something more generalised.  This process not only helps reduce the problem space, it also helps reduce the network's sensitivy to minor changes, in other words to know that two images are of the same object, even though the images are not *exactly* the same.

A somewhat similar process called **Pooling** is also applied to faciliate this *generalisation* even further.  Similar to Artificial Neural Networks, Activation Functions are applied to the data as it moves forward through the network, helping the network decide which neurons will fire, or in other words, helping the network understand which neurons are more or less important for different features, and ultimately which neurons are more or less important for the different output classes.

Over time - as a Convolutional Neural Network trains, it iteratively calculates how well it is predicting on the known classes we pass it (known as the **loss** or **cost**, then heads back through in a process known as **Back Propagation** to update the paramaters within the network, in a way that reduces the error, or in other words, improves the match between predicted outputs and actual outputs.  Over time, it learns to find a good mapping between the input data, and the output classes.

There are many parameters that can be changed within the architecture of a Convolutional Neural Network, as well as clever logic that can be included, all which can affect the predictive accuracy.  We will discuss and put in place many of these below!

___
<br>
# Analysing The Results <a name="causal-impact-results"></a>

<br>
#### Plotting The Results

The *pycausalimpact* library makes plotting the results extremely easy - all done with the single line of code below:

```python

# plot the results
ci.plot()

```
<br>
The resulting plot(s) can be seen below.

<br>
![alt text](/img/posts/causal-impact-results-plot.png "Causal Impact Results Plot")

<br>
To explain what we have in the above image...

The vertical dotted line down the middle of each plot is the date that the Delivery Club membership started.  Everything to the left of this dotted line is the pre-period, and everything to the right of the dotted line is the post-period.

<br>
**Chart 1:  Actual vs. Counterfactual**

The top chart shows the actual data for the impacted group as a black line, in other words the *actual* average daily sales for customers who did go on to sign up to the Delivery Club.  You can also see the counterfactual, which is shown with the blue dotted line.  The purple area around the blue dotted line represent the confidence intervals around the counterfactual - in other words, the range in which the algorithm believes the prediction should fall in.  A wider confidence interval suggests that the model is less sure about it's counterfactual prediction - and this is all taken into account when we look to quantify the actual uplift.

Just eyeing this first chart, it does indeed look like there is some increase in daily average spend for customers who joined the club, over-and-above what the model suggests they would have done, if the club was never in existence.  We will look at the actual numbers for this very soon.

<br>
**Chart 2:  Pointwise Effects**

This second chart shows us, for each day (or data point in general) in our time-series, the *raw differences* between the actual values and the values for the counterfactual.  It is plotting the *differences* from Chart 1.  As an example, if on Day 1 the actual and the counterfactual were the same, this chart would show a value of 0.  If the actual is higher than the counterfactual then we would see a positive value on this chart, and vice versa.  It is essentially showing how far above or below the counterfactual, the actual values are.

What is interesting here is that for the pre-period we see a difference surrounding zero, but in the post period we see mostly positive values mirroring what we saw in Chart 1 where the actual average spend was greater than the counterfactual.

<br>
**Chart 3:  Cumulative Effects**

The bottom chart shows the cumulative uplift over time.  In other words this chart is effectively adding up the Pointwise contributions from the second chart over time.  This is very useful as it helps the viewer get a feel for what the total uplift or difference is at any point in time.

As we would expect based on the other two charts, there does appear to be a cumulative uplift over time.

<br>
#### Interpreting The Numbers

The *pycausalimpact* library also makes interpreting the numbers very easy.  We can get a clean results summary with the following line of code:

```python

# results summary
print(ci.summary())

Posterior Inference {Causal Impact}
                          Average            Cumulative
Actual                    171.33             15762.67
Prediction (s.d.)         121.42 (4.33)      11170.19 (398.51)
95% CI                    [112.79, 129.77]   [10376.65, 11938.77]

Absolute effect (s.d.)    49.92 (4.33)       4592.48 (398.51)
95% CI                    [41.56, 58.54]     [3823.9, 5386.02]

Relative effect (s.d.)    41.11% (3.57%)     41.11% (3.57%)
95% CI                    [34.23%, 48.22%]   [34.23%, 48.22%]

Posterior tail-area probability p: 0.0
Posterior prob. of a causal effect: 100.0%

```
<br>
At the top of the results summary (above) we see that in the post-period the average actual daily sales per customer over the post-period was $171, higher than that of the counterfactual, which was $121.  This counterfactual prediction had 95% confidence intervals of $113 and $130.

Below that we can see the *absolute effect* which is the difference between actual and counterfactual (so the difference between $171 and $121) - and this figure is essentially showing us the average daily *uplift* in sales over the post-period.  We also get the confidence intervals surrounding that effect, and since these do not pass through zero, we can confidently say that there *was* an uplift driven by the Delivery Club.

Below that, we get these same numbers - as percentages.

In the columns on the right of the summary, we see the *cumulative* values for these across the entire post-period, rather than the average per day.

What is amazing about the *pycausalimpact* library is that, with an extra parameter, we can actually get all of this information provided as a written output.

If we put:

```python

# results summary - report
print(ci.summary(output = "report"))

Analysis report {CausalImpact}

During the post-intervention period, the response variable had an average value of approx. 171.33. By contrast, in the absence of an intervention, we would have expected an average response of 121.42.

The 95% interval of this counterfactual prediction is [112.79, 129.77].

Subtracting this prediction from the observed response yields an estimate of the causal effect the intervention had on the response variable. This effect is 49.92 with a 95% interval of [41.56, 58.54]. For a discussion of the significance of this effect, see below.

Summing up the individual data points during the post-intervention period (which can only sometimes be meaningfully interpreted), the response variable had an overall value of 15762.67. By contrast, had the intervention not taken place, we would have expected a sum of 11170.19. The 95% interval of this prediction is [10376.65, 11938.77].

The above results are given in terms of absolute numbers. In relative terms, the response variable showed an increase of +41.11%. The 95% interval of this percentage is [34.23%, 48.22%].

This means that the positive effect observed during the intervention period is statistically significant and unlikely to be due to random fluctuations. It should be noted, however, that the question of whether this increase also bears substantive significance can only be answered by comparing the absolute effect (49.92) to the original goal
of the underlying intervention.

The probability of obtaining this effect by chance is very small (Bayesian one-sided tail-area probability p = 0.0). This means the causal effect can be considered statistically
significant.

```
<br>
So, this is the same information as we saw above, but put into a written report which can go straight to the client.

The high level story of this that, yes, we did see an uplift in sales for those customers that joined the Delivery Club, over and above what we believe they would have spent, had the club not been in existence.  This uplift was deemed to be significantly significant (@ 95%)

___
<br>
# Growth & Next Steps <a name="growth-next-steps"></a>

It would be interesting to look at this pool of customers (both those who did and did not join the Delivery club) and investigate if there were any differences in sales in these time periods *last year* - this would help us understand if any of the uplift we are seeing here is actually the result of seasonality.

It would be interesting to track this uplift over time and see if:

* It continues to grow
* It flattens or returns to normal
* We see any form of uplift pull-forward

It would also be interesting to analyse what it is that is making up this uplift.  Are customers increasing their spend across the same categories - or are they buying into new categories
