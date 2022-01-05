---
layout: post
title: Predicting Customer Loyalty Using Machine Learning
image: "/posts/TesterImagePortfolio1.png"
tags: [Customer Loyalty, Machine Learning, Linear Regression, Decision Tree, Random Forest]
---

# Table of contents

- [Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
    - [Key Definition](#overview-definition)
<br>
<br>
- [Some paragraph](#paragraph1)
    - [Sub paragraph](#subparagraph1)
<br>
<br>
- [Another paragraph](#paragraph2)

## This is the introduction <a name="introduction"></a>
Some introduction text, formatted in heading 2 style

## Some paragraph <a name="paragraph1"></a>
The first paragraph text

### Sub paragraph <a name="subparagraph1"></a>
This is a sub paragraph, formatted in heading 3 style

## Another paragraph <a name="paragraph2"></a>
The second paragraph text

---

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Our client, a grocery retailer, hired a market research consultancy to append market level customer loyalty information to the database.  However, only around 50% of the client's customer base could be tagged, thus the other half did not have this information present.

The overall aim of this work is to accurately predict the *loyalty score* for those customers who could not be tagged, enabling our client a clear understanding of true customer loyalty, regardless of total spend volume - and allowing for more accurate and relevant customer tracking, targeting, and comms.

To achieve this, we looked to build out a predictive model that will find relationships between customer metrics and *loyalty score* for those customers who were tagged, and use this to predict the loyalty score metric for those who were not.
<br>
<br>
### Actions <a name="overview-actions"></a>

We firstly needed to compile the necessary data from tables in the database, gathering key customer metrics that may help predict *loyalty score*, appending on the dependent variable, and separating out those who did and did not have this dependent variable present.

As we are predicting a numeric output, we tested three regression modelling approaches, namely:

* Linear Regression
* Decision Tree
* Random Forest
<br>
<br>

### Results <a name="overview-results"></a>

Our testing found that the Random Forest had the highest predictive accuracy.

<br>
##### Metric 1: Adjusted R-Squared (Test Set)

* Random Forest = 0.955
* Decision Tree = 0.886
* Linear Regression = 0.78

<br>
##### Metric 2: R-Squared (K-Fold Cross Validation, k = 4)

* Random Forest = 0.925
* Decision Tree = 0.871
* Linear Regression = 0.853

As the most important outcome for this project was predictive accuracy, rather than explicitly understanding weighted drivers of prediction, we chose the Random Forest as the model to use for making predictions on the customers who were missing the *loyalty score* metric.
<br>
<br>
### Growth/Next Steps <a name="overview-growth"></a>

While predictive accuracy was relatively high - other modelling approaches could be tested, especially those somewhat similar to Random Forest, for example XGBoost, LightGBM to see if even more accuracy could be gained.

From a data point of view, further variables could be collected, and further feature engineering could be undertaken to ensure that we have as much useful information available for predicting customer loyalty
<br>
<br>
### Key Definition  <a name="overview-definition"></a>

The *loyalty score* metric measures the % of grocery spend (market level) that each customer allocates to the client vs. all of the competitors.  

Example 1: Customer X has a total grocery spend of $100 and all of this is spent with our client. Customer X has a *loyalty score* of 1.0

Example 2: Customer Y has a total grocery spend of $200 but only 20% is spent with our client.  The remaining 80% is spend with competitors.  Customer Y has a *customer loyalty score* of 0.2
<br>
<br>
---

# Data Overview

We will be predicting the *loyalty_score* metric.  This metric exists (for half of the customer base) in the *loyalty_scores* table of the client database.

The key variables hypothesised to predict the missing loyalty scores will come from the client database, namely the *transactions* table, the *customer_details* table, and the *product_areas* table.

Using pandas in Python, we merged these tables together for all customers, creating a single dataset that we can use for modelling.

```ruby

# import required packages
import pandas as pd
import pickle

# import required data tables
loyalty_scores = ...
customer_details = ...
transactions = ...

# merge loyalty score data and customer details data, at customer level
data_for_regression = pd.merge(customer_details, loyalty_scores, how = "left", on = "customer_id")

# aggregate sales data from transactions table
sales_summary = transactions.groupby("customer_id").agg({"sales_cost" : "sum",
                                                         "num_items" : "sum",
                                                         "transaction_id" : "nunique",
                                                         "product_area_id" : "nunique"}).reset_index()

# rename columns for clarity
sales_summary.columns = ["customer_id", "total_sales", "total_items", "transaction_count", "product_area_count"]

# engineer an average basket value column for each customer
sales_summary["average_basket_value"] = sales_summary["total_sales"] / sales_summary["transaction_count"]

# merge the sales summary with the overall customer data
data_for_regression = pd.merge(data_for_regression, sales_summary, how = "inner", on = "customer_id")

# split out data for modelling (loyalty score is present)
regression_modelling = data_for_regression.loc[data_for_regression["customer_loyalty_score"].notna()]

# split out data for scoring post-modelling (loyalty score is missing)
regression_scoring = data_for_regression.loc[data_for_regression["customer_loyalty_score"].isna()]

# for scoring set, drop the loyalty score column (as it is blank/redundant)
regression_scoring.drop(["customer_loyalty_score"], axis = 1, inplace = True)

# save our datasets for future use
pickle.dump(regression_modelling, open("data/abc_regression_modelling.p", "wb"))
pickle.dump(regression_scoring, open("data/abc_regression_scoring.p", "wb"))

```
<br>
After this data pre-processing in Python, we have a dataset for modelling that contains the following fields
<br>
<br>
| **Variable Name** | **Variable Type** | **Description** |
|---|---|---|
| loyalty_score | Dependent | The % of total grocery spend that each customer allocates to ABC Grocery vs. competitors |
| distance_from_store | Independent | "The distance in miles from the customers home address, and the store" |
| gender | Independent | The gender provided by the customer |
| credit_score | Independent | The customers most recent credit score |
| total_sales | Independent | Total spend by the customer in ABC Grocery within the latest 6 months |
| total_items | Independent | Total products purchased by the customer in ABC Grocery within the latest 6 months |
| transaction_count | Independent | Total unique transactions made by the customer in ABC Grocery within the latest 6 months |
| product_area_count | Independent | The number of product areas within ABC Grocery the customers has shopped into within the latest 6 months |
| average_basket_value | Independent | The average spend per transaction for the customer in ABC Grocery within the latest 6 months |

# Modelling Overview

We will


![alt text](/img/posts/linear-regression1.png "Straight Line Equation")

```ruby
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
