---
layout: post
title: Assessing Campaign Performance Using Chi-Square Test For Independence
image: "/posts/ab-testing-title-img.png"
tags: [AB Testing, Hypothesis Testing, Chi-Square, Python]
---

In this project we apply the Chi-Square Test For Independence to assess the performance of two types of mailers that were sent out to promote a new service! 

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [01. Concept Overview](#concept-overview)
- [02. Data Overview & Preparation](#data-overview)
- [03. Applying Chi-Square Test For Independence](#chi-square-application)
- [04. Analysing The Results](#chi-square-results)
- [05. Growth & Next Steps](#growth-next-steps)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Earlier in the year, our client, a grocery retailer, ran a campaign to promote their new "Delivery Club" - an initiative that costs a customer $100 per year for membership, but offers free grocery deliveries rather than the normal cost of $10 per delivery.

For the campaign promoting the club, customers were put randomly into three groups - the first group received a low quality, low cost mailer, the second group received a high quality, high cost mailer, and the third group were a control group, receiving no mailer at all.

The client knows that customers who were contacted, signed up for the Delivery Club at a far higher rate than the control group, but now want to understand if there is a significant difference in signup rate between the cheap mailer and the expensive mailer.  This will allow them to make more informed decisions in the future, with the overall aim of optimising campaign ROI!

<br>
<br>
### Actions <a name="overview-actions"></a>

xxx

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

# Concept Overview  <a name="concept-overview"></a>

<br>
#### A/B Testing

An A/B Test can be described as a randomised experiment containing two groups, A & B, that receive different experiences. Within an A/B Test, we look to understand and measure the response of each group - and the information from this helps drive future business decisions.

Application of A/B testing can range from testing different online ad strategies, different email subject lines when contacting customers, or testing the effect of mailing customers a coupon, vs a control group.  Companies like Amazon are running these tests in an almost never-ending cycle, testing new website features on randomised groups of customers...all with the aim of finding what works best so they can stay ahead of their competition.  Reportedly, Netflix will even test different images for the same movie or show, to different segments of their customer base to see if certain images pull more viewers in.

<br>
#### Hypothesis Testing

A Hypothesis Test is used to assess the plausibility, or likelihood of an assumed viewpoint based on sample data - in other words, a it helps us assess whether a certain view we have about some data is likely to be true or not.

There are many different scenarios we can run Hypothesis Tests on, and they all have slightly different techniques and formulas - however they all have some shared, fundamental steps & logic that underpin how they work.

<br>
**The Null Hypothesis**

In any Hypothesis Test, we start with the Null Hypothesis. The Null Hypothesis is where we state our initial viewpoint, and in statistics, and specifically Hypothesis Testing, our initial viewpoint is always that the result is purely by chance or that there is no relationship or association between two outcomes or groups

<br>
**The Alternate Hypothesis**

The aim of the Hypothesis Test is to look for evidence to support or reject the Null Hypothesis.  If we reject the Null Hypothesis, that would mean we’d be supporting the Alternate Hypothesis.  The Alternate Hypothesis is essentially the opposite viewpoint to the Null Hypothesis - that the result is *not* by chance, or that there *is* a relationship between two outcomes or groups

<br>
**The Acceptance Criteria**

In a Hypothesis Test, before we collect any data or run any numbers - we specify an Acceptance Criteria.  This is a p-value threshold at which we’ll decide to reject or support the null hypothesis.  It is essentially a line we draw in the sand saying "if I was to run this test many many times, what proportion of those times would I want to see different results come out, in order to feel comfortable, or confident that my results are not just some unusual occurrence"

Conventionally, we set our Acceptance Criteria to 0.05 - but this does not have to be the case.  If we need to be more confident that something did not occur through chance alone, we could lower this value down to something much smaller, meaning that we only come to the conclusion that the outcome was special or rare if it’s extremely rare.

So to summarise, in a Hypothesis Test, we test the Null Hypothesis using a p-value and then decide it’s fate based on the Acceptance Criteria.

<br>
**Types Of Hypothesis Test**

There are many different types of Hypothesis Tests, each of which is appropriate for use in differing scenarios - depending on a) the type of data that you’re looking to test and b) the question that you’re asking of that data.

In the case of our task here, where we are looking to understand the difference in sign-up *rate* between two groups - we will utilise the Chi-Square Test For Independence.

<br>
#### Chi-Square Test For Independence

The Chi-Square Test For Independence is a type of Hypothesis Test that assumes observed frequencies for categorical variables will match the expected frequencies.

The *assumption* is the Null Hypothesis, which as discussed above is always the viewpoint that the two groups will be equal.  With the Chi-Square Test For Independence we look to calculate a statistic which, based on the specified Acceptance Criteria will mean we either reject or support this initial assumption.

The *observed frequencies* are the true values that we’ve seen.

The *expected frequencies* are essentially what we would *expect* to see based on all of the data.

**Note:** Another option when comparing "rates" is a test known as the *Z-Test For Proportions*.  While, we could absolutely use this test here, we have chosen the Chi-Square Test For Independence because:

* The resulting test statistic for both tests will be the same
* The Chi-Square Test can be represented using 2x2 tables of data - meaning it can be easier to explain to stakeholders
* The Chi-Square Test can extend out to more than 2 groups - meaning the business can have one consistent approach to measuring signficance

<br>
# Data Overview & Preparation  <a name="data-overview"></a>

In the client database, we have a *campaign_data* table which shows us which customers received each type of "Delivery Club" mailer, which customers were in the control group, and which customers joined the club as a result.

Since Delivery Club membership was open to *all customers* - the control group we have in the *campaign_data* table would help us measure the impact of *contacting* customers but here, we are actually look to measure the overall impact on sales from the Delivery Club itself.  Because of this, we will instead just use customers who did not sign up as the control.  The customers who did not sign up should continue their normal shopping habits after the club went live, and this will help us create the counter-factual for the customers that did sign-up.

In the code below, we:

* Load in the Python libraries we require
* Import the required data from the *transactions* and *campaign_data* tables (3 months prior, 3 months post campaign)
* Aggregate the transactions table from customer/transaction/product area level to customer/date level
* Merge on the signup flag from the *campaign_data* table
* Pivot & aggregate to give us aggregated daily sales by signed-up/did not sign-up groups
* Manoeuvre the data specifically for the pycausalimpact algorithm
* Give our groups some meaningful names, to help with interpretation

<br>
```python

# install the required python libraries
from causalimpact import CausalImpact
import pandas as pd

# import data tables
transactions = ...
campaign_data = ...

# aggregate transaction data to customer, date level
customer_daily_sales = transactions.groupby(["customer_id", "transaction_date"])["sales_cost"].sum().reset_index()

# merge on the signup flag
customer_daily_sales = pd.merge(customer_daily_sales, campaign_data, how = "inner", on = "customer_id")

# pivot the data to aggregate daily sales by signup group
causal_impact_df = customer_daily_sales.pivot_table(index = "transaction_date",
                                                    columns = "signup_flag",
                                                    values = "sales_cost",
                                                    aggfunc = "mean")

# provide a frequency for our DateTimeIndex (avoids a warning message)
causal_impact_df.index.freq = "D"

# ensure the impacted group is in the first column (the library expects this)
causal_impact_df = causal_impact_df[[1,0]]

# rename columns to something lear & meaningful
causal_impact_df.columns = ["member", "non_member"]

```
<br>
A sample of this data (the first 5 days of data) can be seen below:
<br>
<br>

| **transaction_date** | **member** | **non_member** |
|---|---|---|
| 01/04/2020 | 194.49 | 74.46 |
| 02/04/2020 | 185.16 | 75.56 |
| 03/04/2020 | 118.12 | 74.39 |
| 04/04/2020 | 198.53 | 63.00 |
| 05/04/2020 | 145.46 | 72.44 |

<br>
In the DataFrame we have the transaction data, and then a column showing the average daily sales for those who signed up (member) and those who did not (non_member).  This is the required format for applying the algorithm.

<br>
# Applying Chi-Square Test For Independence <a name="chi-square-application"></a>

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

<br>
# Analysing The Results <a name="chi-square-results"></a>

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

<br>
# Growth & Next Steps <a name="growth-next-steps"></a>

It would be interesting to look at this pool of customers (both those who did and did not join the Delivery club) and investigate if there were any differences in sales in these time periods *last year* - this would help us understand if any of the uplift we are seeing here is actually the result of seasonality.

It would be interesting to track this uplift over time and see if:

* It continues to grow
* It flattens or returns to normal
* We see any form of uplift pull-forward

It would also be interesting to analyse what it is that is making up this uplift.  Are customers increasing their spend across the same categories - or are they buying into new categories
