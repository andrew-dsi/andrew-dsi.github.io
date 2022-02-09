---
layout: post
title: Creating An Image Search Engine Using Deep Learning
image: "/posts/dl-search-engine-title-img.png"
tags: [Deep Learning, CNN, Data Science, Computer Vision, Python]
---

In this project we build a Deep Learning based Image Search Engine that will help customers find similar products to ones they want!

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [01. Sample Data Overview](#sample-data-overview)
- [02. Transfer Learning Overview](#transfer-learning-overview)
- [03. Setting Up VGG16](#vgg16-setup)
- [04. Image Preprocessing & Featurisation](#image-preprocessing)
- [05. Execute Search](#execute-search)
- [06. Discussion, Growth & Next Steps](#growth-next-steps)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Our client had been analysing their customer feedback, and one thing in particular came up a number of times.

Their customers are aware that they have a great range of competitively priced products in the clothing section - but have said they are struggling to find the products they are looking for on the website.

They are often buying much more expensive products, and then later finding out that we actually stocked a very similar, but lower-priced alternative.

Based upon our work for them using a Convolutional Neural Network, they want to know if we can build out something that could be applied here.
<br>
<br>
### Actions <a name="overview-actions"></a>

Transfer Learning VGG16
Nearest Neighbours

<br>
<br>

### Results <a name="overview-results"></a>

xxx
xxx

<br>
<br>
### Growth/Next Steps <a name="overview-growth"></a>

xxx
xxx

More products, more categories.  Further analysis of results.  Other distance metrics?

Ways to quantify results?  Customer feedback?  Recommendation Engine metrics...?

Considerations around how it will be implemented (some examples perhaps from the DE/DS section?)

Other pre-trained networks (copy from CNN project)


<br>
<br>

___

# Sample Data Overview  <a name="sample-data-overview"></a>

In the code below, we:

<br>
![alt text](/img/posts/causal-impact-results-plot.png "Causal Impact Results Plot")

<br>
To

___
<br>

# Transfer Learning Overview  <a name="transfer-learning-overview"></a>

<br>
#### Overview

xxx

<br>
#### Nuanced Application

xxx

___
<br>

# Setting Up VGG16  <a name="vgg16-setup"></a>

In the code below, we:

* xxx

<br>
```python



```
<br>
xxx

___
<br>
# Image Preprocessing & Featurisation <a name="image-preprocessing"></a>

In the code below, we specify the start and end dates of the "pre-period" and the start and end dates of the "post-period". We then apply the algorithm by passing in the DataFrame and the specified pre and post period time windows.

The algorithm will model the relationship between members & non-members in the pre-period - and it will use this to create the counterfactual, in other words what it believes would happen to the average daily spend for members in the post-period if no event was to have taken place!

The difference between this counterfactual and the actual data in the post-period will be our "causal impact"

```python

# specify the pre & post periods
pre_period = ["2020-04-01","2020-06-30"]
post_period = ["2020-07-01","2020-09-30"]

# apply the algorithm
ci = CausalImpact(causal_impact_df, pre_period, post_period)

```
<br>
We can use the created object (called ci above) to examine & plot the results.

___
<br>
# Execute Search <a name="execute-search"></a>

<br>
#### Setup

xxx

```python



```
<br>
xxx

<br>
#### Preprocess & Featurise

xxx

```python



```
<br>
xxx

<br>
#### Locate Most Similar Images

xxx

```python



```
<br>
xxx

<br>
#### Plot Search Results

xxx

```python



```
<br>
xxx

___
<br>
# Discussion,Growth & Next Steps <a name="growth-next-steps"></a>

xxx
