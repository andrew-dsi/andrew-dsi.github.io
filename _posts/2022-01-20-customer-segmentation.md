---
layout: post
title: The "You Are What You Eat" Customer Segmentation
image: "/posts/clustering-title-img.png"
tags: [Customer Segmentation, Machine Learning, Clustering, Python]
---

In this project we use k-means clustering to segment up the customer base in order to increase business understanding, and to enhance the relevancy of targeted messaging & customer communications.

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [01. Data Overview](#data-overview)
- [02. K-Means](#kmeans-title)
    - [Concept Overview](#kmeans-overview)
    - [Data Preprocessing](#kmeans-preprocessing)
    - [Finding A Good Value For K](#kmeans-k-value)
    - [Model Fitting](#kmeans-model-fitting)
    - [Appending Clusters To Customers](#kmeans-append-clusters)
    - [Segment Profiling](#kmeans-cluster-profiling)
- [03. Application](#kmeans-application)
- [04. Growth & Next Steps](#growth-next-steps)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

The Senior Management team from our client, a supermarket chain, are disagreeing about how customers are shopping, and how lifestyle choices may affect which food areas customers are shopping into, or more interestingly, not shopping into.

They have asked us to use data, and Machine Learning to help segment up their customers based upon their engagement with each of the major food categories - aiding business understanding of the customer base, and to enhance the relevancy of targeted messaging & customer communications.

<br>
<br>
### Actions <a name="overview-actions"></a>

We firstly needed to compile the necessary data from sevaral tables in the database, namely the *transactions* table and the *product_areas* table.  We joined together the relevant information using Pandas, and then aggregated the transactional data across product areas, from the most recent six month to a customer level.

As a starting point, we test & apply k-means clustering for this task.  We need to apply some data pre-processing, most importantly feature scaling to ensure all variables exist on the same scale - a very important consideration for distance based algorithms such as k-means.

As k-means is an *unsupervised learning* approach, in other words there are no labels - we use a process known as *Within Cluster Sum of Squares (WCSS)* to understand what a "good" number of clusters or segments is.

Based upon this, we apply the k-means algorithm onto the product area data, append the clusters to our customer base, and then profile the resulting customer segments to understand what the differentiating factors were!
<br>
<br>

### Results <a name="overview-results"></a>

xxxxx

<br>
<br>
### Growth/Next Steps <a name="overview-growth"></a>

While predictive accuracy was relatively high - other modelling approaches could be tested, especially those somewhat similar to Random Forest, for example XGBoost, LightGBM to see if even more accuracy could be gained.

From a data point of view, further variables could be collected, and further feature engineering could be undertaken to ensure that we have as much useful information available for predicting customer loyalty
<br>
<br>

___

# Data Overview  <a name="data-overview"></a>

We are primarily looking to discover segments of customers based upon their transactions within *food* based product areas so we will need to only select those.

In the code below, we:

* Import the required python packages & libraries
* Import the tables from the database
* Merge the tables to tag on *product_area_name* which only exists in the *product_areas* table
* Drop the non-food categories
* Aggregate the sales data for each product area, at customer level
* Pivot the data to get it into the right format for clustering
* Change the values from raw dollars, into a percentage of spend for each customer (to ensure each customer is comparable)

<br>
```python

# import required Python packages
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

# import tables from database
transactions = ...
product_areas = ...

# merge product_area_name on
transactions = pd.merge(transactions, product_areas, how = "inner", on = "product_area_id")

# drop the non-food category
transactions.drop(transactions[transactions["product_area_name"] == "Non-Food"].index, inplace = True)

# aggregate sales at customer level (by product area)
transaction_summary = transactions.groupby(["customer_id", "product_area_name"])["sales_cost"].sum().reset_index()

# pivot data to place product areas as columns
transaction_summary_pivot = transactions.pivot_table(index = "customer_id",
                                                    columns = "product_area_name",
                                                    values = "sales_cost",
                                                    aggfunc = "sum",
                                                    fill_value = 0,
                                                    margins = True,
                                                    margins_name = "Total").rename_axis(None,axis = 1)

# transform sales into % sales
transaction_summary_pivot = transaction_summary_pivot.div(transaction_summary_pivot["Total"], axis = 0)

# drop the "total" column as we don't need that for clustering
data_for_clustering = transaction_summary_pivot.drop(["Total"], axis = 1)

```
<br>

After the data pre-processing using Pandas, we have a dataset for clustering that looks like the below sample:
<br>
<br>

| **customer_id** | **dairy** | **fruit** | **meat** | **vegetables** |
|---|---|---|---|---|
| 2 | 0.246 | 0.198 | 0.394 | 0.162  |
| 3 | 0.142 | 0.233 | 0.528 | 0.097  |
| 4 | 0.341 | 0.245 | 0.272 | 0.142  |
| 5 | 0.213 | 0.250 | 0.430 | 0.107  |
| 6 | 0.180 | 0.178 | 0.546 | 0.095  |
| 7 | 0.000 | 0.517 | 0.000 | 0.483  |

<br>
The data is at customer level, and we have a column for each of the highest level food product areas.  Within each of those we have the *percentage* of sales that each customer allocated to that product area over the past six months.

<br>
# K-Means <a name="kmeans-title"></a>

<br>
### Concept Overview <a name="kmeans-overview"></a>

K-Means is an *unsupervised learning* algorithm, meaning that it does not look to predict known labels or values, but instead looks to isolate patterns within unlabelled data.

The algorithm works in a way where it partitions data-points into distinct groups (clusters) based upon their *similarity* to each other.

This similarity is most often the eucliedean (straight-line) distance between data-points in n-dimensional space.  Each variable that is included lies on one of the dimensions in space.

The number of distinct groups (clusters) is determined by the value that is set for "k".

The algorithm does this by iterating over four key steps, namely:

1. It selects "k" random points in space (these points are known as centroids)
2. It then assigns each of the data points to the nearest centroid (based upon euclidean distance)
3. It then repositions the centroids to the *mean* dimension values of it's cluster
4. It then reassigns each data-point to the nearest centroid

Steps 3 & 4 continue to iterate until no data-points are reassigned to a closer centroid.

<br>
### Data Preprocessing <a name="kmeans-preprocessing"></a>

There are three vital preprocessing steps for k-means, namely:

* Missing values in the data
* The effect of outliers
* Feature Scaling

<br>
##### Missing Values

Missing values can cause issues for k-means, as the algorithm won't know where to plot those data-points along the dimension where the value is not present.  If we have observations with missing values, the most common options are to either remove the observations, or to use an imputer to fill-in or to estimate what those value might be.

As we aggregated our data for each customer, we actually don't suffer from missing values so we don't need to deal with that here.

<br>
##### Outliers

As k-means is a distance based algorithm, outliers can cause problems. The main issue we face is when we come to scale our input variables, a very important step for a distance based algorithm.

We don’t want any variables to be “bunched up” due to a single outlier value, as this will make it hard to compare their values to the other input variables. We should always investigate outliers rigorously - however in our case where we're dealing with percentages, we thankfully don't face this issue!

<br>
##### Feature Scaling

Again, as k-means is a distance based algorithm, in other words it is reliant on an understanding of how similar or different data points are across different dimensions in n-dimensional space, the application of Feature Scaling is extremely important.

Feature Scaling is where we force the values from different columns to exist on the same scale, in order to enchance the learning capabilities of the model. There are two common approaches for this, Standardisation, and Normalisation.

Standardisation rescales data to have a mean of 0, and a standard deviation of 1 - meaning most datapoints will most often fall between values of around -4 and +4.

Normalisation rescales datapoints so that they exist in a range between 0 and 1.

For k-means clustering, either approach is going to be *far better* than using no scaling at all.  Here, we will look to apply normalisation as this will ensure all variables will end up having the same range, fixed between 0 and 1, and therefore the k-means algorithm can judge each variable in the same context.  Standardisation *can* result in different ranges, variable to variable, and this is not so useful (although this isn't explcitly true in all scenarios).

Another reason for choosing Normalisation over Standardisation is that our scaled data will *all* exist between 0 and 1, and these will then be compatible with any categorical variables that we have encoded as 1’s and 0’s (although we don't have any variables of this type in our task here).

In our specific task here, we are using percentages, so our values are _already_ spread between 0 and 1.  We will still apply normalisation for the following reason.  One of the product areas might commonly make up a large proportion of customer sales, and this may end up dominating the clustering space.  If we normalise all of our variables, even product areas that make up smaller volumes, will be spread proportionately between 0 and 1!

The below code uses the in-built MinMaxScaler functionality from scikit-learn to apply Normalisation to all of our variables.  The reason we create a new object (here called data_for_clustering_scaled) is that we want to use the scaled data for clustering, but when profiling the clusters later on, we may want to use the actual percentages as this may make more intuitive business sense, so it's good to have both options available!

```python

# create our scaler object
scale_norm = MinMaxScaler()

# normalise the data
data_for_clustering_scaled = pd.DataFrame(scale_norm.fit_transform(data_for_clustering), columns = data_for_clustering.columns)

```

<br>
### Finding A Good Value For k <a name="kmeans-k-value"></a>

At this point here, our data is ready to be fed into the k-means clustering algorithm.  Before that however, we want to understand what number of clusters we want the data split into.

In the world of unsupervised learning, there is no *right or wrong* value for this - it really depends on the data you are dealing with, as well as the unique scenario you're utilising the algorithm for.  From our client, having a very high number of clusters might not be appropriate as it would be too hard for the business to understand the nuance of each in a way where they can apply the right strategies.

Finding the "right" value for k, can feel more like art than science, but there are some data driven approaches that can help us!  

The approach we will utilise here is known as *Within Cluster Sum of Squares (WCSS)* which measures the sum of the squared euclidean distances that data points lie from their closest centroid.  WCSS can help us understand the point where adding *more clusters* provides little extra benefit in terms of separating our data.

By default, the k-means algorithm within scikit-learn will use k = 8 meaning that it will look to split the data into eight distinct clusters.  We want to find a better value that fits our data, and our task!

In the code below we will test multiple values for k, and plot how this WCSS metric changes.  As we increase the value for k (in other words, as we increase the number or centroids or clusters) the WCSS value will always decrease.  However, these decreases will get smaller and smaller each time we add another centroid and we are looking for a point where this decrease is quite prominent *before* this point of diminishiing returns....

```python

# instantiate our model object
clf = LogisticRegression(random_state = 42, max_iter = 1000)

# fit our model using our training & test sets
clf.fit(X_train, y_train)

```

<br>
### Model Fitting <a name="kmeans-model-fitting"></a>

Instantiating and training our Logistic Regression model is done using the below code.  We use the *random_state* parameter to ensure reproducible results, meaning any refinements can be compared to past results.  We also specify *max_iter = 1000* to allow the solver more attempts at finding an optimal regression line, as the default value of 100 was not enough.

```python

# instantiate our model object
clf = LogisticRegression(random_state = 42, max_iter = 1000)

# fit our model using our training & test sets
clf.fit(X_train, y_train)

```

<br>
### Append Clusters To Customers <a name="kmeans-append-clusters"></a>

##### Predict On The Test Set

To assess how well our model is predicting on new data - we use the trained model object (here called *clf*) and ask it to predict the *signup_flag* variable for the test set.

In the code below we create one object to hold the binary 1/0 predictions, and another to hold the actual prediction probabilities for the positive class.

```python

# predict on the test set
y_pred_class = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:,1]

```

<br>
##### Confusion Matrix

A Confusion Matrix provides us a visual way to understand how our predictions match up against the actual values for those test set observations.

The below code creates the Confusion Matrix using the *confusion_matrix* functionality from within scikit-learn and then plots it using matplotlib.

```python

# create the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_class)

# plot the confusion matrix
plt.style.use("seaborn-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i, j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j, i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()

```

<br>
![alt text](/img/posts/log-reg-confusion-matrix.png "Logistic Regression Confusion Matrix")

<br>
The aim is to have a high proportion of observations falling into the top left cell (predicted non-signup and actual non-signup) and the bottom right cell (predicted signup and actual signup).

Since the proportion of signups in our data was around 30:70 we will next analyse not only Classification Accuracy, but also Precision, Recall, and F1-Score which will help us assess how well our model has performed in reality.

<br>
##### Classification Performance Metrics
<br>
**Classification Accuracy**

Classification Accuracy is a metric that tells us *of all predicted observations, what proportion did we correctly classify*.  This is very intuitive, but when dealing with imbalanced classes, can be misleading.  

An example of this could be a rare disease. A model with a 98% Classification Accuracy on might appear like a fantastic result, but if our data contained 98% of patients *without* the disease, and 2% *with* the disease - then a 98% Classification Accuracy could be obtained simply by predicting that *no one* has the disease - which wouldn't be a great model in the real world.  Luckily, there are other metrics which can help us!

In this example of the rare disease, we could define Classification Accuracy as *of all predicted patients, what proportion did we correctly classify as either having the disease, or not having the disease*

<br>
**Precision & Recall**

Precision is a metric that tells us *of all observations that were predicted as positive, how many actually were positive*

Keeping with the rare disease example, Precision would tell us *of all patients we predicted to have the disease, how many actually did*

Recall is a metric that tells us *of all positive observations, how many did we predict as positive*

Again, referring to the rare disease example, Recall would tell us *of all patients who actually had the disease, how many did we correctly predict*

The tricky thing about Precision & Recall is that it is impossible to optimise both - it's a zero-sum game.  If you try to increase Precision, Recall decreases, and vice versa.  Sometimes however it will make more sense to try and elevate one of them, in spite of the other.  In the case of our rare-disease prediction like we've used in our example, perhaps it would be more important to optimise for Recall as we want to classify as many positive cases as possible.  In saying this however, we don't want to just classify every patient as having the disease, as that isn't a great outcome either!

So - there is one more metric we will discuss & calculate, which is actually a *combination* of both...

<br>
**F1 Score**

F1-Score is a metric that essentially "combines" both Precision & Recall.  Technically speaking, it is the harmonic mean of these two metrics.  A good, or high, F1-Score comes when there is a balance between Precision & Recall, rather than a disparity between them.

Overall, optimising your model for F1-Score means that you'll get a model that is working well for both positive & negative classifications rather than skewed towards one or the other.  To return to the rare disease predictions, a high F1-Score would mean we've got a good balance between successfully predicting the disease when it's present, and not predicting cases where it's not present.

Using all of these metrics in combination gives a really good overview of the performance of a classification model, and gives us an understanding of the different scenarios & considerations!

<br>
In the code below, we utilise in-built functionality from scikit-learn to calculate these four metrics.

```python

# classification accuracy
accuracy_score(y_test, y_pred_class)

# precision
precision_score(y_test, y_pred_class)

# recall
recall_score(y_test, y_pred_class)

# f1-score
f1_score(y_test, y_pred_class)

```
<br>
Running this code gives us:

* Classification Accuracy = **0.866** meaning we correctly predicted the class of 86.6% of test set observations
* Precision = **0.784** meaning that for our *predicted* delivery club signups, we were correct 78.4% of the time
* Recall = **0.69** meaning that of all *actual* delivery club signups, we predicted correctly 69% of the time
* F1-Score = **0.734** 

Since our data is *somewhat* imbalanced, looking at these metrics rather than just Classification Accuracy on it's own - is a good idea, and gives us a much better understanding of what our predictions mean!  We will use these same metrics when applying other models for this task, and can compare how they stack up.

<br>
### Segment Profiling <a name="kmeans-cluster-profiling"></a>

By default, most pre-built classification models & algorithms will just use a 50% probability to discern between a positive class prediction (delivery club signup) and a negative class prediction (delivery club non-signup).

Just because 50% is the default threshold *does not mean* it is the best one for our task.

Here, we will test many potential classification thresholds, and plot the Precision, Recall & F1-Score, and find an optimal solution!

<br>
```python

# set up the list of thresholds to loop through
thresholds = np.arange(0, 1, 0.01)

# create empty lists to append the results to
precision_scores = []
recall_scores = []
f1_scores = []

# loop through each threshold - fit the model - append the results
for threshold in thresholds:
    
    pred_class = (y_pred_prob >= threshold) * 1
    
    precision = precision_score(y_test, pred_class, zero_division = 0)
    precision_scores.append(precision)
    
    recall = recall_score(y_test, pred_class)
    recall_scores.append(recall)
    
    f1 = f1_score(y_test, pred_class)
    f1_scores.append(f1)
    
# extract the optimal f1-score (and it's index)
max_f1 = max(f1_scores)
max_f1_idx = f1_scores.index(max_f1)

```
<br>

Now we have run this, we can use the below code to plot the results!

<br>
```python

# plot the results
plt.style.use("seaborn-poster")
plt.plot(thresholds, precision_scores, label = "Precision", linestyle = "--")
plt.plot(thresholds, recall_scores, label = "Recall", linestyle = "--")
plt.plot(thresholds, f1_scores, label = "F1", linewidth = 5)
plt.title(f"Finding the Optimal Threshold for Classification Model \n Max F1: {round(max_f1,2)} (Threshold = {round(thresholds[max_f1_idx],2)})")
plt.xlabel("Threshold")
plt.ylabel("Assessment Score")
plt.legend(loc = "lower left")
plt.tight_layout()
plt.show()

```
<br>
![alt text](/img/posts/log-reg-optimal-threshold-plot.png "Logistic Regression Optimal Threshold Plot")

<br>
Along the x-axis of the above plot we have the different classification thresholds that were testing.  Along the y-axis we have the performance score for each of our three metrics.  As per the legend, we have Precision as a blue dotted line, Recall as an orange dotted line, and F1-Score as a thick green line.  You can see the interesting "zero-sum" relationship between Precision & Recall *and* you can see that the point where Precision & Recall meet is where F1-Score is maximised.

As you can see at the top of the plot, the optimal F1-Score for this model 0.78 and this is obtained at a classification threshold of 0.44.  This is higher than the F1-Score of 0.734 that we achieved at the default classification threshold of 0.50!

<br>
# Application <a name="kmeans-application"></a>

We now have a model object, and a the required pre-processing steps to use this model for the next *delivery club* campaign.  When this is ready to launch we can aggregate the neccessary customer information and pass it through, obtaining predicted probabilities for each customer signing up.

Based upon this, we can work with the client to discuss where their budget can stretch to, and contact only the customers with a high propensity to join.  This will drastically reduce marketing costs, and result in a much improved ROI.

<br>
# Growth & Next Steps <a name="growth-next-steps"></a>

While predictive accuracy was relatively high - other modelling approaches could be tested, especially those somewhat similar to Random Forest, for example XGBoost, LightGBM to see if even more accuracy could be gained.

We could even look to tune the hyperparameters of the Random Forest, notably regularisation parameters such as tree depth, as well as potentially training on a higher number of Decision Trees in the Random Forest.

From a data point of view, further variables could be collected, and further feature engineering could be undertaken to ensure that we have as much useful information available for predicting customer loyalty
