---
title: "Bookstore Promotional Activity : Predicted Sales Evaluation"
last_modified_at: 2018-11-30T17:00:00-00:00
categories:
  - data-science
tags:
  - hackathon
  - machine learning
classes:
    - wide
---

<style>
figcaption {
  text-align: center;
}
</style>

<figure style="width: 800px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/bookstore/bookstore.jpg" alt="">
  <figcaption class="align-center">Pic for illustration purposes only (Credits: nytimes.com)
</figcaption>
</figure>

## Introduction

The overall goal of this project involved predicting whether a given customer will respond to a specific promotional activity, and how much they will spend if they do respond. Almost all of the given data relies on time, in some variety, since the best way to predict future purchases is to observe overall purchase history among customers. We focused on the importance of the “time factor” in numerous ways in our analysis, hypothesizing that partitioning data based on date ordered would prove extremely beneficial. Therefore, we concentrated on creating new variables
dependent on the time frame of an order, hoping to find a “sweet spot”of time since the last order that would make a customer most likely to buy again. Our section on data cleaning and
exploratory data analysis details our methods for creating these time-sensitive variables, along with their interactions, in the hopes of creating a successful predictive model.
Another goal of ours involved exhaustively searching the data for any relevant variables,
knowing that we could rely on stepwise selection to reduce the number of predictors to the most significant subset. This included delving into the complex dataset of category variables, which separated orders by category of book ordered. With the assistance of random forest to determine the most significant of these categories, we wished to include these categories in our analysis. This category analysis is detailed at the beginning of our model fitting section. After narrowing down which category variables are the most significant, we moved on to building the core of our analysis: our predictive models. Our model fitting section discusses our methods to create optimized logistic regression and multiple linear regression models, including
outlier removal, model diagnostics such as Cook’s distance and VIF calculation, and stepwise selection. Finally, we discuss the accuracy of our models, both statistically and financially.

## Data Cleaning and Exploratory Data Analysis

Our goal regarding data cleaning and exploratory data analysis involved identifying,
combining and partitioning variables that we hypothesize to be potentially significant predictors.
We first endeavoured to pinpoint significant information, if any, contained in the category
category amount variables. No correlations were very high, with the highest correlations between
variables. These variables, containing data regarding amount and frequency of orders separated
by book category, proved cumbersome due to their sheer number. Rather than abandoning these
variables, we looked into ways to reduce their dimensions by examining the relationships
between categories. To begin, we looked at the correlation matrix (Appendix 1) between the
certain category variables being slightly less than 0.50, but in the amount correlation matrix, we
were able to identify some categories which seemed to be related, with a correlation greater than
0.3 considered as significant. It was observed that the following categories were intercorrelated:
Fiction, Cartoon, Art, History, Travel Guides, Hobby, Contemporary History, and Nature. In our
feature engineering steps, we created a variable Mgroup, adding together the amount values for
these categories, and the variable Fgroup, adding together the frequency values for these
categories. To further test these relationships, we ran a random forest model including all the
category variables, which identified one group of category variables that are most related.
Observations regarding individual orders and order pricing, namely that sometimes a
single book can cost upwards of $1000, caused us to question the significance of the “amount”
variable representing the total amount spent by a single customer summed across all order data.
Therefore, we created a separate variable, qty, representing the total number of books bought by
a single customer summed across all order data. Dividing the qty variable by time on file yielded
a useful interaction that we later included in both the logistic and linear regression models.
Additionally, we created a variable called amount upon quantity that created another metric
comparing dollar amount and quantity of books by dividing amount by qty. Other interaction
variables created included: dividing amount by total number of orders (“frequency” variable),
dividing qty by frequency, and dividing frequency by time on file. Please refer to Appendix 2 for
a summary on all engineered features.
For dealing with outliers, we created three criteria based on some of the interaction
variables mentioned above. We removed observations where:
1. amount divided by the total number of orders (amtuponorders) exceeded $1000
2. the total number of books per total number of orders (qtyuponorders) exceeded 40
3. amount divided by total number of books ordered (amtuponqty) exceeded $600.
Overall these three criteria only removed 0.0355% (12 observations) of the total data, while
improving the fit of our two models on the training data. While we cannot be certain that these
data are incorrect, our goal was to create the most accurate model while not mispredicting large
values in the test data, and these criteria best struck that balance. The scatterplots for the
variables both before and after outlier removal are illustrated in Appendix 3. Later on, for our
multiple linear regression model, we used Cook’s Distance to identify a few more outliers.
As stated in the introduction, much of the data relies on time to chart consumer behavior,
and we endeavoured to capture the effect of the time an order is placed on the response variable.
While the variable “recency” charts the number of days since the most recent order a customer
has placed, the order date variable records the date of all orders a customer has placed. We
utilized information about both the most recent order and the total orders, per customer, to build
a number of novel features in the hopes of illuminating relationships between time and targeted
amount. We separated customers by timeframe of order, partitioning customers into groups
based on who had ordered in the last 1, 3, 6, and 12 months. Summary statistics for each group,
by customer, were calculated, namely quantity of books ordered in this time frame, designated
“qty”, and total dollar amount, designated “price”. Thus, we were able to separate the two
variables discussed in the previous paragraph, amount and qty, into a few time-frame partitions.
When considering the total number of orders per customer, we were able to calculate the
percentage of the total amount purchased, per customer, that occured in these time frames.
Therefore, we were able to observe sale trends for each customer over the past year.
When organizing customers based on order date in this manner, we decided to order time
on file in a similar way. Partitioning customers based on time on file in the last 1, 3, 6, and 12
months gave us a separation between newer and older customers. During this analysis of time on
file, we discovered an interesting observation about a small subset of the customers. In the
training set, 87 individuals have a time on file value of zero; similarly, 271 individuals in the test
set have a time on file value of zero. These customers are brand new to the online store
database; therefore, they have no previous order data and so would be nearly impossible to create
accurate predictions for. We were unable to remove these observations due to 52.9% of the new
customers in the training set being responders. Since the overall percentage of responders in the
dataset is 3.93%, removing the new customers would remove a significant portion of the
responders. We therefore decided to manually impute predicted values for these observations.
The imputation process will be detailed in the next section, as it differed for the logistic and the
linear regression models.