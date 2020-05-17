---
title: "Luxury European Hotels: Predicting Guest Review Scores Based on Hotel Reviews
"
last_modified_at: 2019-03-15T17:00:00-00:00
categories:
  - data-science
tags:
  - Predictive Analytics
  - Supervised Learning
  - Neural Networks
  - Gradient Boosted Trees
  - Random Forest
  - MARS
  - Machine learning 
classes:
    - wide
header:
  teaser: "assets/images/hotel_review/hotelimage.jpg"
---

<style>
figcaption {
  text-align: center;
}
</style>

The goal of this analysis was to predict the score, on a scale from 1 (lowest) to 10 (highest) that a reviewer will give after a stay at a luxury hotel in one of six European cities.

<figure style="width: 800px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/hotel_review/hotelimage.jpg" alt="">
  <figcaption class="align-center"> Hotel Novo Plaza Pera in Istanbul (For illustration purposes only)
</figcaption>
</figure>

In order to generate our predictions, we examined a set of 515,738 reviews with data about the hotel, the reviewer and some elements of the review besides the score, which we augmented by parsing text tags, transforming features, and pulling in outside weather data. We fit the resulting data with several supervised learning models to generate a predicted score for each review. Gradient boosting trees provided the best fit for our data, accounting for **42.8%** of the variance of reviews in our test set. We found that, unsurprisingly, the ratio of positive words to negative words in the review was the strongest predictor of the reviewer score. Other important predictors were the distance from the city center, the total length of the review and the high and low temperatures for the day of the hotel visit. Our primary conclusion was that factors available outside of the review provided limited predictive power as to how a reviewer would respond.

## INTRODUCTION

As our world becomes increasingly digital, more and more consumers are relying on online reviews to help them decide what to buy, where to travel, and where to stay. Especially in the hotel review industry, these online customer reviews can make or break a business. According to surveys conducted by Trip Advisor and Trust You (siteminder.com): 

1.	88% of travellers filter out hotels with an average star rating below three [on a scale of one to five]
2.	32% eliminated those with a rating below four
3.	96% consider reviews important when researching a hotel
4.	79% will read between six and 12 reviews before making a purchase decision
5.	Four out of five believe a hotel that responds to reviews cares more about its customers
6.	85% agree that a thoughtful response to a review will improve their impression of the hotel

We have a dataset from [Kaggle](https://www.kaggle.com/), originally scraped from Booking.com, with information on 515,738 reviews covering 1,493 unique luxury hotels in Europe. We will be predicting the rating that each user will give a particular hotel based on a stay. We will identify how different attributes of each hotel and each customer impact the review score, and identify whether a particular type of customer has a propensity to provide a positive or a negative review. Our analysis can help a hotel identify its loyal group of customers and the areas of service in which it excels. This can help drive future rebranding, marketing and customer service strategy and also help identify gaps where a hotel could potentially improve.

## DATA CLEANING AND EXPLORATORY DATA ANALYSIS

### DATA OVERVIEW

The original dataset included 17 different variables. Each reviewer leaves certain pros and cons for a particular hotel and a rating for the hotel from 1 (lowest) to 10 (highest). We also had additional information, which fell into three broad categories: hotel specific information, guest specific information and stay or review specific information. The variables are listed below:

**Hotel Specific Information:**
| Field | Description|
|-------|-------------|
| Hotel_Name	| Name of the hotel |
| Hotel_Address	| Address of the hotel |
| lat	| Latitude of the hotel location |
| lng	| Longitude of the hotel location |
| Total_Number_of_Reviews	| Total number of valid reviews that a hotel has been given on booking.com |
| Additional_Number_of_Scoring	| There are also some guests who just made a scoring on the service rather than a review - This number indicates how many valid scores a hotel has without a review |
| Average_Score	| Average score of the hotel, calculated based on the latest comment in the last year |




## TEAM MEMBERS
- [Michel Leroy](https://www.linkedin.com/in/sarah-michel-leroy/)
- [Dhansree Suraj](https://www.linkedin.com/in/dhansree-suraj/)
- [Tony Colucci](https://www.linkedin.com/in/anthony-colucci-710659112/)
- [Tanya Tandon](https://www.linkedin.com/in/tanya-tandon/)
- [Saurabh Annadate](https://www.linkedin.com/in/saurabhannadate93/)

## LINKS
- Dataset: https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe
