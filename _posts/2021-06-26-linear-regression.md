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
pickle.dump(regression_modelling, open("data/customer_loyalty_modelling.p", "wb"))
pickle.dump(regression_scoring, open("data/customer_loyalty_scoring.p", "wb"))

```
<br>
After this data pre-processing in Python, we have a dataset for modelling that contains the following fields...
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

<br>
# Modelling Overview

We will build a model that looks to accurately predict the “loyalty_score” metric for those customers that were able to be tagged, based upon the customer metrics listed above.

If that can be achieved, we can use this model to predict the customer loyalty score for the customers that were unable to be tagged by the agency.

As we are predicting a numeric output, we tested three regression modelling approaches, namely:

* Linear Regression
* Decision Tree
* Random Forest

The code, details, and results can be found below.

<br>
# Linear Regression

We utlise the scikit-learn library within Python to model our data using Linear Regression. The code sections below are broken up into 4 key sections:

* Data Import
* Data Preprocessing
* Model Training
* Performance Assessment
<br>
### Data Import <a name="linreg-import"></a>

Since we saved our modelling data as a pickle file, we import it.  We ensure we remove the id column, and we also ensure our data is shuffled.

```ruby

#########################################################################
# import required packages
#########################################################################

import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV

#########################################################################
# import modelling data
#########################################################################

# import
data_for_model = pickle.load(open("data/customer_loyalty_modelling.p", "rb"))

# drop uneccessary columns
data_for_model.drop("customer_id", axis = 1, inplace = True)

# shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)

```
<br>
### Data Preprocessing <a name="linreg-preprocessing"></a>

For Linear Regression we have certain data preprocessing steps that need to be addressed, including:

* Missing values in the data
* The effect of outliers
* Encoding categorical variables to numeric form
* Multicollinearity & Feature Selection

<br>
##### Missing Values

The number of missing values in the data was extremely low, so instead of applying any imputation (i.e. mean, most common value) we will just remove those rows

```ruby

# remove rows where values are missing
data_for_model.isna().sum()
data_for_model.dropna(how = "any", inplace = True)

```

<br>
##### Outliers

The ability for a Linear Regression model to generalise well across *all* data can be hampered if there are outliers present.  There is no right or wrong way to deal with outliers, but it is always something worth very careful consideration - just because a value is high or low, does not necessarily mean it should not be there!

In this code section, we use **.describe()** from Pandas to investigate the spread of values for each of our predictors.  The results of this can be seen in the table below.

<br>

| **metric** | **distance_from_store** | **credit_score** | **total_sales** | **total_items** | **transaction_count** | **product_area_count** | **average_basket_value** |
|---|---|---|---|---|---|---|---|
| mean | 2.02 | 0.60 | 1846.50 | 278.30 | 44.93 | 4.31 | 36.78 |
| std | 2.57 | 0.10 | 1767.83 | 214.24 | 21.25 | 0.73 | 19.34 |
| min | 0.00 | 0.26 | 45.95 | 10.00 | 4.00 | 2.00 | 9.34 |
| 25% | 0.71 | 0.53 | 942.07 | 201.00 | 41.00 | 4.00 | 22.41 |
| 50% | 1.65 | 0.59 | 1471.49 | 258.50 | 50.00 | 4.00 | 30.37 |
| 75% | 2.91 | 0.66 | 2104.73 | 318.50 | 53.00 | 5.00 | 47.21 |
| max | 44.37 | 0.88 | 9878.76 | 1187.00 | 109.00 | 5.00 | 102.34 |

<br>
Based on this investigation, we see some *max* column values for several variables to be much higher than the *median* value.

This is for columns *distance_from_store*, *total_sales*, and *total_items*

For example, the median *distance_to_store* is 1.645 miles, but the maximum is over 44 miles!

Because of this, we apply some outlier removal in order to facilitate generalisation across the full dataset.

We do this using the "boxplot approach" where we remove any rows where the values within those columns are outside of the interquartile range multiplied by 2.

<br>
```ruby

outlier_investigation = data_for_model.describe()
outlier_columns = ["distance_from_store", "total_sales", "total_items"]

# boxplot approach
for column in outlier_columns:
    
    lower_quartile = data_for_model[column].quantile(0.25)
    upper_quartile = data_for_model[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 2
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    outliers = data_for_model[(data_for_model[column] < min_border) | (data_for_model[column] > max_border)].index
    print(f"{len(outliers)} outliers detected in column {column}")
    
    data_for_model.drop(outliers, inplace = True)

```

<br>
##### Split Out Data For Modelling

In the next code block we do two things, we firstly split our data into an **X** object which contains only the predictor variables, and a **y** object that contains only our dependent variable.

Once we have done this, we split our data into training and test sets to ensure we can fairly validate the accuracy of the predictions on data that was not used in training.

```ruby

# split data into X and y objects for modelling
X = data_for_model.drop(["customer_loyalty_score"], axis = 1)
y = data_for_model["customer_loyalty_score"]

# split out training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

```

<br>
##### Categorical Predictor Variables

In the next code

![alt text](/img/posts/linear-regression1.png "Straight Line Equation")

![alt text](/img/posts/chart-image1.png "Image")

The above image is just linked to the actual file in my Github, but I could also link to images online, using the URL!

A line break, like this one below - helps me make sense of what I'm reading, especially when I've had so much coffee that my vision goes a little blurry

---

I could also add things to my project like links, tables, quotes, and HTML blocks - but I'm starting to get a cracking headache.  Must be coffee time.

![alt text](/img/posts/chart-image.png "Straight Line Equation")
