---
layout: post
title: Understanding Alcohol Product Relationships Using Association Rule Learning
image: "/posts/association-title-img.jpg"
tags: [Association Rule Learning, Python]
---

In this project we use Association Rule Learning to analyse the transactional relationships & dependencies between products in the alcohol section of a grocery store.

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [01. Data Overview](#data-overview)
- [02. Apriori Overview](#apriori-overview)
- [03. Data Preparation](#apriori-data-prep)
- [04. Applying The Apriori Algorithm](#apriori-fit)
- [04. Interpreting The Results](#apriori-results)
- [05. Growth & Next Steps](#growth-next-steps)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Our client is looking to re-jig the alcohol section within their store.  Customers are often complaining that they can't find the products they want, and are also wanting recommendations about which other products to try.  On top of this, their marketing team would like to start running "bundled" promotions as this has worked well in other areas of the store - but need guidance with selecting which products to put together.

They have provided us a sample of 3,500 alcohol transactions - our task is fairly open - to see if we can find solutions or insights that might help the business address the aforementioned problems!

<br>
<br>
### Actions <a name="overview-actions"></a>

Based upon the tasks at hand - we apply Association Rule Learning, specifically *Apriori* to examine & analyse the strength of relationship between different products within the transactional data.

We firstly needed to bring in the sample data, and get it into the right format for the Apriori algorithm to deal with.

From there we apply the Apriori algorithm to provide us with several different relationship metrics, namely:

* Support
* Confidence
* Expected Confidence
* Lift

These metrics examine product relationships in different ways, so we utilise each to put forward ideas that address each of the tasks at hand

<br>
<br>

### Results <a name="overview-results"></a>

xxx

<br>
<br>
### Growth/Next Steps <a name="overview-growth"></a>

xxx

<br>
<br>

___

# Data Overview  <a name="data-overview"></a>

Our dataset contains only 356 customers, but 102 columns.

In the code below, we:

* Import the required python packages & libraries
* Import the data from the database
* Drop the ID column for each customer
* Shuffle the dataset
* Analyse the class balance between album buyers, and non album buyers

<br>
```python

# import required Python packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# import data
data_for_model = ...

# drop the id column
data_for_model.drop("user_id", axis = 1, inplace = True)

# shuffle the data
data_for_model = shuffle(data_for_model, random_state = 42)

# analyse the class balance
data_for_model["purchased_album"].value_counts(normalize = True)

```
<br>

From the last step in the above code, we see that 53% of customers in our sample did purchase Ed's last album, and 47% did not. Since this is evenly balanced, we can most likely rely solely on *Classification Accuracy* when assessing the performance of the classification model later on.

After these steps, we have a dataset that looks like the below sample (not all columns shown):
<br>
<br>

| **purchased_album** | **artist1** | **artist2** | **artist3** | **artist4** | **artist5** | **artist6** | **artist7** | **…** |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0278 | 0 | 0 | 0 | 0 | 0.0036 | 0.0002 | … |
| 1 | 0 | 0 | 0.0367 | 0.0053 | 0 | 0 | 0.0367 | … |
| 1 | 0.0184 | 0 | 0 | 0 | 0 | 0 | 0 | … |
| 0 | 0.0017 | 0.0226 | 0 | 0 | 0 | 0 | 0 | … |
| 1 | 0.0002 | 0 | 0 | 0 | 0 | 0 | 0 | … |
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | … |
| 1 | 0.0042 | 0 | 0 | 0 | 0 | 0 | 0 | … |
| 0 | 0 | 0 | 0.0002 | 0 | 0 | 0 | 0 | … |
| 1 | 0 | 0 | 0 | 0 | 0.1759 | 0 | 0 | … |
| 1 | 0.0001 | 0 | 0.0001 | 0 | 0 | 0 | 0 | … |
| 1 | 0 | 0 | 0 | 0.0555 | 0 | 0.0003 | 0 | … |
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | … |
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | … |

<br>
The data is at customer level.  We have a binary column showing whether the customer purchased the prior album or not, and following that 100 columns containing the percentage of historical listening time allocated to each artist.  We do not know the names of these artists.

From the above sample, we can also see the sparsity of the data, customers do not listen to all artists and therefore many of the values are 0.

<br>
# Apriori Overview  <a name="data-overview"></a>

Association Rule Learning is an approach that discovers the strength of relationships between different data-points.  It is commonly utilised to understand which products are frequently (or infrequently) purchased together.

In a business sense this can provide some really interesting, and useful information that can help optimise:

* Product Arrangement/Placement (making the customer journey more efficient)
* Product Recommendations (customers who purchased product A also purchased product B)
* Bundled Discounts (which products should/should not be put together)

One powerful, intuitive, and commonly used algorithm for Association Rule Learning is **Apriori**.

In Apriori there are four key metrics, namely:

* Support
* Confidence
* Expected Confidence
* Lift

Each of these metrics help us understand items, and their relationship with other items in their own way.

##### Support

Support is extremely intuitive, it simply tells us the percentage of all transactions that contain *both* Item A and Item B.  To calculate this we’d just count up the transactions that include both items, and divide this by the total number of transactions.

You can think of Support as a baseline metric that helps us understand how common or popular this particular *pair* of items is.

##### Confidence

Confidence takes us a little bit further than Support, and looks more explcitly at the *relationship* between the two items.

It asks "of all transactions that *included item A*, what proportion also included item B?"  

In other words, here we are counting up the number of transactions that contained *both items A and B* and then rather than dividing by *all transactions* like we did for Support, we instead divide this by the *total number of transactions that contained item A*.

A high score for Confidence can mean a strong product relationship - but not always!  When one of the items is very popular we can get an inflated score.  To help us regulate this, we can look at two further metrics, Expected Confidence and Lift!

##### Expected Confidence

Expected Confidence is quite simple, it is the percentage of *all transactions* that *contained item B*.

This is important as it provides indication of what the Confidence *would be* if there were no relationship between the items.  We can use Expected Confidence, along with Confidence to calculate our final (and most powerful) metric - Lift!

##### Lift

Lift is the factor by which the Confidence, exceeds the Expected Confidence.  In other words, Lift tells us how likely item B is purchased *when item A is purchased*, while *controlling* for how popular item B is.

We calculate Lift by dividing Confidence by Expected Confidence.


<br>
# Data Preparation  <a name="pca-data-prep"></a>

<br>
##### Split Out Data For Modelling

In the next code block we do two things, we firstly split our data into an X object which contains only the predictor variables, and a y object that contains only our dependent variable.

Once we have done this, we split our data into training and test sets to ensure we can fairly validate the accuracy of the predictions on data that was not used in training. In this case, we have allocated 80% of the data for training, and the remaining 20% for validation. We make sure to add in the stratify parameter to ensure that both our training and test sets have the same proportion of customers who did, and did not, sign up for the delivery club - meaning we can be more confident in our assessment of predictive performance.

```python

# split data into X and y objects for modelling
X = data_for_model.drop(["purchased_album"], axis = 1)
y = data_for_model["purchased_album"]

# split out training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

```

<br>
##### Feature Scaling

Feature Scaling is extremely important when applying PCA - it means that the algorithm can successfully "judge" the correlations between the variables and effectively create the principal compenents for us.  The general consensus is to apply Standardisation rather than Normalisation.

The below code uses the in-built StandardScaler functionality from scikit-learn to apply Standardisation to all of our variables.

In the code, we use *fit_transform* for the training set, but only *transform* to the test set. This means the standardisation logic will learn and apply the “rules” from the training data, but only apply them to the test data. This is important in order to avoid data leakage where the test set learns information about the training data, and means we can’t fully trust model performance metrics!

```python

# create our scaler object
scale_standard = StandardScaler()

# standardise the data
X_train = scale_standard.fit_transform(X_train)
X_test = scale_standard.transform(X_test)

```

<br>
# Fitting PCA <a name="pca-fit"></a>

We firstly apply PCA to our training set without limiting the algorithm to any particular number of components, in other words we're not explicitly reducing the feature space at this point.

Allowing all components to be created here allows us to examine & plot the percentage of explained variance for each, and assess which solution might work best for our task.

In the code below we instantiate our PCA object, and then fit it to our training set.

```python

# instantiate our PCA object (no limit on components)
pca = PCA(n_components = None,  random_state = 42)

# fit to our training data
pca.fit(X_train)

```

<br>
# Analysis Of Explained Variance <a name="pca-variance"></a>

There is no right or wrong number of components to use - this is something that we need to decide based upon the scenario we're working in.  We know we want to reduce the number of features, but we need to trade this off with the amount of information we lose.

In the following code, we extract this information from the prior step where we fit the PCA object to our training data.  We extract the variance for each component, and we do the same again, but for the *cumulative* variance.  Will will assess & plot both of these in the next step.

```python

# explained variance across components
explained_variance = pca.explained_variance_ratio_

# explained variance across components (cumulative)
explained_variance_cumulative = pca.explained_variance_ratio_.cumsum()

```

<br>
In the following code, we create two plots - one for the variance of each principal component, and one for the cumulative variance.

```python

num_vars_list = list(range(1,101))
plt.figure(figsize=(16,9))

# plot the variance explained by each component
plt.subplot(2,1,1)
plt.bar(num_vars_list,explained_variance)
plt.title("Variance across Principal Components")
plt.xlabel("Number of Components")
plt.ylabel("% Variance")
plt.tight_layout()

# plot the cumulative variance
plt.subplot(2,1,2)
plt.plot(num_vars_list,explained_variance_cumulative)
plt.title("Cumulative Variance across Principal Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative % Variance")
plt.tight_layout()
plt.show()

```
<br>
As we can see in the top plot, PCA works in a way where the first component holds the most variance, and each subsequent component holds less and less.

The second plot shows this as a cumulative measure - and we can how many components we would need remain in order to keep any amount of variance from the original feature set. 

<br>
![alt text](/img/posts/pca-variance-plots.png "PCA Variance by Component")

<br>
Based upon the cumulative plot above, we can see that we could keep 75% of the variance from the original feature set with only around 25 components, in other words with only a quarter of the number of features we can still hold onto around three-quarters of the information.

<br>
# Applying our PCA solution <a name="pca-application"></a>

Now we've run our analysis of variance by component - we can apply our PCA solution.

In the code below - we *re-instantiate* our PCA object, this time specifying that we want the number of components that will keep 75% of the initial variance.

We then apply this solution to both our training set (using fit_transform) and our test set (using transform only).

Finally - based on this 75% threshold, we confirm the number of components that this leaves us with.

```python

# re-instantiate our PCA object (keeping 75% of variance)
pca = PCA(n_components = 0.75,  random_state = 42)

# fit to our data
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# check the number of components
print(pca.n_components_)

```

<br>
Turns out we were almost correct from looking at our chart - we will retain 75% of the information from our initial feature set, with only 24 principal components.

Our X_train and X_test objects now contain 24 columns, each representing one of the principal components - we can see a sample of X_train below:

| **0** | **1** | **2** | **3** | **4** | **5** | **6** | **…** |
|---|---|---|---|---|---|---|---|
| -0.402194 | -0.756999 | 0.219247 | -0.0995449 | 0.0527621 | 0.0968236 | -0.0500932 | … |
| -0.360072 | -1.13108 | 0.403249 | -0.573797 | -0.18079 | -0.305604 | -1.33653 | … |
| 10.6929 | -0.866574 | 0.711987 | 0.168807 | -0.333284 | 0.558677 | 0.861932 | … |
| -0.47788 | -0.688505 | 0.0876652 | -0.0656084 | -0.0842425 | 1.06402 | 0.309337 | … |
| -0.258285 | -0.738503 | 0.158456 | -0.0864722 | -0.0696632 | 1.79555 | 0.583046 | … |
| -0.440366 | -0.564226 | 0.0734247 | -0.0372701 | -0.0331369 | 0.204862 | 0.188869 | … |
| -0.56328 | -1.22408 | 1.05047 | -0.931397 | -0.353803 | -0.565929 | -2.4482 | … |
| -0.282545 | -0.379863 | 0.302378 | -0.0382711 | 0.133327 | 0.135512 | 0.131 | … |
| -0.460647 | -0.610939 | 0.085221 | -0.0560837 | 0.00254932 | 0.534791 | 0.251593 | … |
| … | … | … | … | … | … | … | … |

<br>
Here, column "0" represents the first component, column "1" represents the second component, and so on.  This are the input variables we will feed into our classification model to predict which customers purchased Ed Sheeran's last album!

<br>
# Classification Model <a name="pca-classification"></a>

##### Training The Classifier

To start with, we will simply apply a Random Forest Classifier to see if it is possible to predict based upon our set of 24 components.  

In the code below we instantiate the Random Forest using the default parameters, and then we fit this to our data.

```python

# instantiate our model object
clf = RandomForestClassifier(random_state = 42)

# fit our model using our training & test sets
clf.fit(X_train, y_train)

```
<br>
##### Classification Performance

In the code below we use the trained classifier to predict on the test set - and we run a simple analysis for the classification accuracy for the predictions vs. actuals.

```python

# predict on the test set
y_pred_class = clf.predict(X_test)

# assess the classification accuracy
accuracy_score(y_test, y_pred_class)

```
<br>
The result of this is a **93%** classification accuracy, in other words, using a classifier trained on 24 principal components we were able to accurately predict which test set customers purchased Ed Sheeran's last album, with an accuracy of 93%.

<br>
# Application <a name="kmeans-application"></a>

Based upon this proof-of-concept, we could go back to the client and recommend that they purchase some up to date listening data.  We would could apply PCA to this, create the components, and predict which customers are likely to buy Ed's *next* album.

<br>
# Growth & Next Steps <a name="growth-next-steps"></a>

We only tested one type of classifier here (Random Forest) - it would be worthwhile testing others.  We also only used the default classifier hyperparameters - we would want to optimise these.

Here, we selected 24 components based upon the fact this accounted for 75% of the variance of the initial feature set.  We would instead look to search for the optimal number of components to use based upon classification accuracy.
