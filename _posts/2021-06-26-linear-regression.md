---
layout: post
title: Predicting Loyalty Using Linear Regression
image: "/posts/TesterImagePortfolio1.png"
tags: [Machine Learning, Linear Regression]
---

In this project we will look to predict loyalty scores for customers that an agency could not tag.  The model is based upon the customers that do have loyalty scores assigned, and the relationship between that variable and various customer metrics.

```
# Import required Python packages

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

# Import sample data

my_df = pd.read_csv("data/sample_data_regression.csv")

# Split data into input and output objects

X = my_df.drop(["output"], axis = 1)
y = my_df["output"]

# Split data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Instantiate our model object

regressor = LinearRegression()

# Train our model

regressor.fit(X_train, y_train)

# Assess model accuracy

y_pred = regressor.predict(X_test)
r2_score(y_test, y_pred)
```
![alt text](/img/posts/linear-regression1.png "Straight Line Equation")

Here is an **unordered list** showing some things I love about Python

* For my work
    * Data Analysis
    * Data Visualisation
    * Machine Learning
* For fun
    * Deep Learning
    * Computer Vision
    * Projects about coffee

Here is an _ordered list_ showing some things I love about coffee

1. The smell
    1. Especially in the morning, but also at all times of the day!
2. The taste
3. The fact I can run the 100m in approx. 9 seconds after having 4 cups in quick succession

I love Python & Coffee so much, here is that picture from the top of my project AGAIN, but this time, in the BODY of my project!

![alt text](/img/posts/chart-image1.png "Image")

The above image is just linked to the actual file in my Github, but I could also link to images online, using the URL!

A line break, like this one below - helps me make sense of what I'm reading, especially when I've had so much coffee that my vision goes a little blurry

---

I could also add things to my project like links, tables, quotes, and HTML blocks - but I'm starting to get a cracking headache.  Must be coffee time.

![alt text](/img/posts/chart-image.png "Straight Line Equation")
