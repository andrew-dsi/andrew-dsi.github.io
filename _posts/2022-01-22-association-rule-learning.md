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

We firstly installed the apyori package, which contains all of the required functionality for this task.

We then needed to bring in the sample data, and get it into the right format for the Apriori algorithm to deal with.

From there we apply the Apriori algorithm to provide us with several different relationship metrics, namely:

* Support
* Confidence
* Expected Confidence
* Lift

These metrics examine product relationships in different ways, so we utilise each to put forward ideas that address each of the tasks at hand.  You can read more about these metrics, and the Apriori algorithm in the relevant section below.

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

Our initial dataset contains 3,500 transactions, each of which shows the alcohol products that were present in that transaction.  

In the code below, we import Pandas, as well as the apriori algorithm from the apyori library, and we bring the raw data into Python.
<br>
```python

# import required Python packages
import pandas as pd
from apyori import apriori

# import the sample data
alcohol_transactions = pd.read_csv("data/sample_data_apriori.csv")

```
<br>

A sample of this data (the first 10 transactions) can be seen below:
<br>
<br>

| **transaction_id** | **product1** | **product2** | **product3** | **product4** | **product5** | **…** |
|---|---|---|---|---|---|---|
| 1 | Premium Lager | Iberia | … |  |  | ... |
| 2 | Sparkling | Premium Lager | Premium Cider | Own Label | Italy White | … |
| 3 | Small Sizes White | Small Sizes Red | Sherry Spanish | No/Low Alc Cider | Cooking Wine | … |
| 4 | White Uk | Sherry Spanish | Port | Italian White | Italian Red | … |
| 5 | Premium Lager | Over-Ice Cider | French White South | French Rose | Cocktails/Liqueurs | … |
| 6 | Kosher Red | … |  |  |  | ... |
| 7 | Own Label | Italy White | Australian Red | … |  | ... |
| 8 | Brandy/Cognac | … |  |  |  | ... |
| 9 | Small Sizes White | Bottled Ale | … |  |  | ... |
| 10 | White Uk | Spirits Mixers | Sparkling | German | Australian Red | … |
| … | … | … | … | … | … | … |

<br>
To explain this data, *Transaction 1* (the first row) contained two products, Premium Lager, and Iberia.  As there were only two products in this transaction, the remaining columns are blank.

Transaction 2 (the second row) contained nine products (not all shown in the snippet).  The first nine columns for this row are therefore populated, followed by blank values.

For our sample data, the maximum number of unique products was 45, meaning the table of data had a total of 46 columns (45 for products + transaction_id).

The *apyori* library that we are using does not want the data in this format, it instead wants it passed in as a *list of lists* so we will need to modify it.  The code and logic for this can be found in the Data Preparation section below.

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

<br>
##### Support

Support is extremely intuitive, it simply tells us the percentage of all transactions that contain *both* Item A and Item B.  To calculate this we’d just count up the transactions that include both items, and divide this by the total number of transactions.

You can think of Support as a baseline metric that helps us understand how common or popular this particular *pair* of items is.

<br>
##### Confidence

Confidence takes us a little bit further than Support, and looks more explcitly at the *relationship* between the two items.

It asks "of all transactions that *included item A*, what proportion also included item B?"  

In other words, here we are counting up the number of transactions that contained *both items A and B* and then rather than dividing by *all transactions* like we did for Support, we instead divide this by the *total number of transactions that contained item A*.

A high score for Confidence can mean a strong product relationship - but not always!  When one of the items is very popular we can get an inflated score.  To help us regulate this, we can look at two further metrics, Expected Confidence and Lift!

<br>
##### Expected Confidence

Expected Confidence is quite simple, it is the percentage of *all transactions* that *contained item B*.

This is important as it provides indication of what the Confidence *would be* if there were no relationship between the items.  We can use Expected Confidence, along with Confidence to calculate our final (and most powerful) metric - Lift!

<br>
##### Lift

Lift is the factor by which the Confidence, exceeds the Expected Confidence.  In other words, Lift tells us how likely item B is purchased *when item A is purchased*, while *controlling* for how popular item B is.

We calculate Lift by dividing Confidence by Expected Confidence.

A Lift score *greater than 1* indicates that items A & B appear together *more often* than expected, and conversely a Lift score *less then 1* indicates that items A & B appear together *less often* than expected.

<br>
##### In Practice

While above we're just discussing two products (Item A & Item B) - in reality this score would be calculated between *all* pairs of products, and we could then sort these by Lift score (for example) and see exactly what the strongest or weakest relationships were - and this information would guide our decisions regarding product layout, recommendations for customers, or promotions.

<br>
##### An Important Consideration

Something to consider when assessing the results of Apriori is that, Item/Product relationships that have a *high Lift score* but also have a *low Support score* should be interpreted with caution!

In other words, if we sorted all Item relationships by descending Lift score, the one that comes out on top might initially seem very impressive and it may appear that there is a very strong relationship between the two items.  Always take into account the Support metric - it could be that this relationship is only taking place by chance due to the rarity of the item set.

<br>
# Data Preparation  <a name="apriori-data-prep"></a>

As mentioned in the Data Overview section above, the *apyori* library that we are using does not want the data in table format, it instead wants it passed in as a *list of lists* so we will need to modify it here.  

In the code below, we:

* Remove the ID column as it is not required
* Iterate over the DataFrame, appending each transaction to a list, and appending those to a master list
* Print out the first 10 lists from the master list

<br>
```python

# drop ID column
alcohol_transactions.drop("transaction_id", axis = 1, inplace = True)

# modify data for apriori algorithm
transactions_list = []
for index, row in alcohol_transactions.iterrows():
    transaction = list(row.dropna())
    transactions_list.append(transaction)
    
# print out first 10 lists from master list
print(transactions_list[:10])

[['Premium Lager', 'Iberia'],
 ['Sparkling', 'Premium Lager', 'Premium Cider', 'Own Label', 'Italy White', 'Italian White', 'Italian Red', 'French Red', 'Bottled Ale'],
 ['Small Sizes White', 'Small Sizes Red', 'Sherry Spanish', 'No/Low Alc Cider', 'Cooking Wine', 'Cocktails/Liqueurs', 'Bottled Ale'],
 ['White Uk', 'Sherry Spanish', 'Port', 'Italian White', 'Italian Red'],
 ['Premium Lager', 'Over-Ice Cider', 'French White South', 'French Rose', 'Cocktails/Liqueurs', 'Bottled Ale'],
 ['Kosher Red'],
 ['Own Label', 'Italy White', 'Australian Red'],
 ['Brandy/Cognac'],
 ['Small Sizes White', 'Bottled Ale'],
 ['White Uk', 'Spirits Mixers', 'Sparkling', 'German', 'Australian Red', 'American Red']]

```
<br>

As you can see from the print statement, each transaction (row) from the initial DataFrame is now contained within a list, all making up the master list.

<br>
# Applying The Apriori Algorithm <a name="apriori-fit"></a>

In the code below we apply the apriori algorithm from the apyori library.

This algorithm allows us to specify the association rules that we want.  We set:

* A minimum *Support* of 0.003 to eliminate very rare product sets
* A minimum *Confidence* of 0.2
* A minimum *Lift* of 3 to ensure we're only focusing on product sets with strong relationships
* A minimum & maximum length of 2 meaning we're only focusing on product *pairs* rather than larger sets

```python

# apply the apriori algorthm and specify required parameters
apriori_rules = apriori(transactions_list,
                        min_support = 0.003,
                        min_confidence = 0.2,
                        min_lift = 3,
                        min_length = 2,
                        max_length = 2)

# convert the output to a list
apriori_rules = list(apriori_rules)

# print out the first element
apriori_rules[0]

RelationRecord(items=frozenset({'America White', 'American Rose'}), support=0.020745724698626296, ordered_statistics=[OrderedStatistic(items_base=frozenset({'American Rose'}), items_add=frozenset({'America White'}), confidence=0.5323741007194245, lift=3.997849299507762)])

```
<br>
The output from the algorithm is in the form of a generator. We covert this to a list as this is easier to manipulate & analyse.  

Based upon the parameters we set when applying the algorithm, we get 132 product pairs.  We print out the first element from the list to see what the output looks like, and while this contains all the key information we need - to make it easier to analyse (and more accessible & useable for stakeholders) - in the next code snippet, we extract the key elements and use list comprehension to re-work this data to exist as a Pandas DataFrame.

```python

# extract each piece of information
product1 = [list(rule[2][0][0])[0] for rule in apriori_rules]
product2 = [list(rule[2][0][1])[0] for rule in apriori_rules]
support = [rule[1] for rule in apriori_rules]
confidence = [rule[2][0][2] for rule in apriori_rules]
lift = [rule[2][0][3] for rule in apriori_rules]

# compile into a single dataframe
apriori_rules_df = pd.DataFrame({"product1" : product1,
                                 "product2" : product2,
                                 "support" : support,
                                 "confidence": confidence,
                                 "lift" : lift})

```
<br>
A sample of this data (the first 5 product pairs - not in any order) can be seen below:
<br>
<br>

| **product1** | **product2** | **support** | **confidence** | **lift** |
|---|---|---|---|---|
| American Rose | America White | 0.021 | 0.532 | 3.998 |
| America White | American White | 0.054 | 0.408 | 3.597 |
| Australian Rose | America White | 0.005 | 0.486 | 3.653 |
| Low Alcohol A.C | America White | 0.003 | 0.462 | 3.466 |
| American Rose | American Red | 0.016 | 0.403 | 3.575 |
| … | … | … | … | … |

<br>
In the DataFrame we have the two products in the pair, and then the three key metrics; Support, Confidence, and Lift. 

<br>
# Interpreting The Results <a name="apriori-results"></a>

Now we have our data in a useable format - let's look at the product pairs with the *strongest* relationships - we can do this by sorting our Lift column, in descending order.

```python

# sort pairs by descending Lift
apriori_rules_df.sort_values(by = "lift", ascending = False, inplace = True)

```

<br>
xxxxxxxxx

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
