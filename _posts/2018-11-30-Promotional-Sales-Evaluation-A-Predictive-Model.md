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


## Introduction

We were given a scenario where an online bookstore decided to reach out to customers via email as a promotional activity to increase sales. The goal of our project was to predict whether a given customer will respond to the email, and how much they will spend on buying books if they do respond.

<figure style="width: 800px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/bookstore/bookstore.jpg" alt="">
  <figcaption class="align-center">Inside the Last Bookstore in downtown L.A. (Credit : Joe Leavenworth)
</figcaption>
</figure>

For our analysis, we were provided past history of all the bookstore's customers and their response to an earlier conducted promotional activity.  Our assumption was that the past buying history of a customer was indicative of whether he/she will respond to this promotional activity or not. Several feature indicating buying patterns (recency, frequency, amount) were constructed and evaluated to understand the customer behaviour. We focused on the importance of the “time factor” in numerous ways in our analysis, hypothesizing that partitioning data based on date ordered would prove extremely beneficial. Our section on data cleaning and exploratory data analysis details our methods for creating these time-sensitive variables, along with their interactions, in the hopes of creating a successful predictive model.

Another goal of ours involved exhaustively searching the large feature space for any indicative features, knowing that we could rely on stepwise selection to reduce the number of predictors to the most significant subset. We had information on the category of books (Art, music, fiction etc.) ordered by the customers. Constructing features centered around each of these categories led us to have > 60 features in our initial model. In order to separate the grain from chaff, we utilized the random forest algorithm for selecting the most impactful variables. This analysis is detailed at the beginning of our model fitting section. After narrowing down which category variables are the most significant, we moved on to building the core of our analysis: our predictive models. Our model fitting section discusses our methods to create optimized logistic regression and multiple linear regression models, including outlier removal, model diagnostics such as Cook’s distance and VIF calculation, and stepwise selection. Finally, we discuss the accuracy of our models, both statistically and financially.

## Data Cleaning and Exploratory Data Analysis

Our goal regarding data cleaning and exploratory data analysis involved identifying, combining and partitioning variables that we hypothesize to be potentially significant predictors. We first endeavoured to pinpoint significant information, if any, contained in the category variables. These variables, containing data regarding amount and frequency of orders separated by book category, proved cumbersome due to their sheer number. Rather than abandoning these variables, we looked into ways to reduce their dimensions by examining the relationships between categories. To begin, we looked at the correlation matrix (Figure 1) between the category amount variables.

<figure style="width: 800px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/bookstore/Fig1.PNG" alt="">
  <figcaption class="align-center">Figure 1: Correlation matrix between the category amount variables
</figcaption>
</figure>

No correlations were very high, with the highest correlations between certain category variables being slightly less than 0.50, but in the amount correlation matrix, we were able to identify some categories which seemed to be related, with a correlation greater than 0.3 considered as significant. It was observed that the following categories were intercorrelated: Fiction, Cartoon, Art, History, Travel Guides, Hobby, Contemporary History, and Nature. In our feature engineering steps, we created a variable Mgroup, adding together the amount values for these categories, and the variable Fgroup, adding together the frequency values for these categories.

Observations regarding individual orders and order pricing, namely that sometimes a single book can cost upwards of $1000, caused us to question the significance of the “amount”
variable representing the total amount spent by a single customer summed across all order data. Therefore, we created a separate variable, qty, representing the total number of books bought by
a single customer summed across all order data. Dividing the qty variable by time on file yielded a useful interaction that we later included in both the logistic and linear regression models.
Additionally, we created a variable called amount upon quantity that created another metric comparing dollar amount and quantity of books by dividing amount by qty. Other interaction
variables created included: dividing amount by total number of orders (“frequency” variable), dividing qty by frequency, and dividing frequency by time on file.

For dealing with outliers, we created three criteria based on some of the interaction variables mentioned above. We removed observations where:
1. Amount divided by the total number of orders (amtuponorders) exceeded $1000
2. The total number of books per total number of orders (qtyuponorders) exceeded 40
3. Amount divided by total number of books ordered (amtuponqty) exceeded $600.

Overall these three criteria only removed 0.0355% (12 observations) of the total data, while improving the fit of our two models on the training data. While we cannot be certain that these data are incorrect, our goal was to create the most accurate model while not mispredicting large values in the test data, and these criteria best struck that balance. The scatterplots for the variables both before and after outlier removal are illustrated in Figure 2. Later on, for our multiple linear regression model, we used Cook’s Distance to identify a few more outliers.

<figure style="width: 800px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/bookstore/Fig2_1.PNG" alt="">
</figure>

<figure style="width: 800px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/bookstore/Fig2_2.PNG" alt="">
</figure>

<figure style="width: 800px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/bookstore/Fig2_3.PNG" alt="">
  <figcaption class="align-center">Figure 2: Before (Left) and After (Right) outlier removal
</figcaption>
</figure>
