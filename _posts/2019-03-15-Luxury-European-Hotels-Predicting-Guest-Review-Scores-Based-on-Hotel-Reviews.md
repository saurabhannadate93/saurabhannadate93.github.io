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
  - TF-IDF
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

| Field | Description |
|-------|-------------|
| Hotel_Name	| Name of the hotel |
| Hotel_Address	| Address of the hotel |
| lat	| Latitude of the hotel location |
| lng	| Longitude of the hotel location |
| Total_Number_of_Reviews	| Total number of valid reviews that a hotel has been given on booking.com |
| Additional_Number_of_Scoring	| There are also some guests who just made a scoring on the service rather than a review - This number indicates how many valid scores a hotel has without a review |
| Average_Score	| Average score of the hotel, calculated based on the latest comment in the last year |

**Reviewer Specific Information**

| Field | Description |
|-------|-------------|
| Reviewer_Nationality	| Nationality of the reviewer |
| Total_Number_of_Reviews_Reviewer_Has_Given	| Number of reviews the reviewer has given in the past on Booking.com |

**Stay Specific Information**

| Field | Description |
|-------|-------------|
| Reviewer_Score	| Score the reviewer has given to the hotel, based on his/her experience (this is the response variable in our models) |
| Negative_Review	| What the reviewer wrote in the “negative” or “cons” section of the review | 
| Review_Total_Negative_Word_Counts	| Total number of words in the negative review section |
| Positive_Review	| What the reviewer wrote in the “positive” or “pros” section of the review |
| Review_Total_Positive_Word_Counts	| Total number of words in the positive review section |
| Tags	| Tags reviewer gave the hotel |
| Review_Date	| Date when reviewer posted the corresponding review |
| days_since_review	| Duration between the review date and scrape date |

## EXPLORATORY DATA ANALYSIS AND FEATURE ENGINEERING

As shown in Figure 1 below, our response variable was fairly well distributed and covered a wide range of the rating scale, with a minimum score of 2.5 and a maximum of 10. 

<figure style="width: 800px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/hotel_review/Fig1.PNG" alt="">
  <figcaption class="align-center"> Figure 1: Review Score distribution
</figcaption>
</figure>

Based on the variables available to us, we focused on 4 distinct feature groups: features of the guest, the distinct perks of each hotel, outside factors that could affect the customer’s stay, and finally, features of the review itself.

To determine a hotel’s “usual” guest, we used the customer tags related to each stay. The tags included information such as whether the trip was for leisure or business, whether the guest was traveling with a family, the length of stay, and the type of room. We started by aggregating these at a hotel level to assign tags to the hotels. We only aggregated the tags from highly rated reviews so the tags would be indicative of that hotel’s particular expertise. We then used TF-IDF (Term Frequency - Inverse Document Frequency) to determine the importance of each tag, by using each tag as a term and the hotel’s aggregated tags as the document. We ended up with 7 broad categories: trip type, room type, luxury room type, view type, access type, time of stay, and whether the guest was provided free services. For each of these categories, every hotel was assigned its top tag, i.e. tags for which it received great reviews. Once we had the hotel tags, we created a compatibility score variable that captured hotel and customer compatibility. We assigned the customer tags to each of the 7 broad categories. We then compared the customer tags to the hotel tags for each of the categories, adding one to the compatibility score for each tag that matched.

Another feature we created about each guest was whether a customer was a tourist. We defined “tourist” as a customer who was not from the same country as the hotel. If the hotel’s address matched the customer’s nationality, the customer was not marked as a tourist. We also analyzed the home country of each guest. As we had 227 unique countries, we decided to aggregate these into 18 sub-regions. Figure 2 shows the percentage of guests from each sub-region.

<figure style="width: 800px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/hotel_review/Fig2.PNG" alt="">
  <figcaption class="align-center"> Figure 2: Guest Origination Country Distribution
</figcaption>
</figure>

Finally, we used the number of reviews given by a reviewer as a feature as well. We log transformed this variable as this was heavily right skewed, as you can see from the histogram below.

<figure style="width: 800px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/hotel_review/Fig3_1.PNG" alt="">
</figure>

<figure style="width: 800px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/hotel_review/Fig3_2.PNG" alt="">
  <figcaption class="align-center"> Figure 3: Distribution for the total number of reviews given by a guest (above) and log transformed (below)
</figcaption>
</figure>

Secondly, we analyzed the perks of each hotel. The first variable we created was the hotel city, which we extracted from the address. The hotel reviews take place in the following six cities: 

| Vienna, Austria | Paris, France | Milan, Italy | Amsterdam, Netherlands | Barcelona, Spain | London, United Kingdom |
|---|---|---|---|---|---|
| 38,939 | 59,928 | 37,207 | 57,214 | 60,149 | 262,301 |
| 8% | 12% | 7% | 11% | 12% | 51% |

We also thought it would be interesting to have a feature which notes the distance from the city centre. A consistent characteristic of European cities is a central city centre that is also a public transportation hub. We hypothesize that hotels closer to the city centre would receive a higher rating as this would improve the guest’s overall experience due to accessibility and proximity to tourist destinations. 

Additionally, we focused on external factors affecting the quality of a stay. We hypothesized that the weather could affect whether a stay was pleasant. To create these features, we pulled historical weather data for each review using the Dark Sky API (darksky.net/dev), based on the hotel city and the review date (as a proxy for the stay date). We created features from the high and low temperature. We also used the summary to find the most common types of weather, and created a weather summary feature with 7 factors: breezy, clear, cloudy, foggy, humid, rain, and snow. 

Finally, we focused on features of the review itself. Our original data included the text and word count of the user review split out into positive and negative sections. We calculated a sum of the word counts from these positive and negative review sections then log-transformed that aggregate count, to correct for the heavily right-skewed distribution. In order to summarize the ratio of positive to negative words in the review, we generated the pct_positive variable, which is the percent of total review words in the positive section. We weighted this towards the middle for reviews with fewer than 50 words, under the assumption that people who wrote fewer words potentially felt less strongly than those who wrote more words. We weighted percent positive using the following formula:

*pct_positive = (positive word count ÷50) - (total word count ÷100)  + 0.5* {: style="text-align: center;"}





## TEAM MEMBERS
- [Michel Leroy](https://www.linkedin.com/in/sarah-michel-leroy/)
- [Dhansree Suraj](https://www.linkedin.com/in/dhansree-suraj/)
- [Tony Colucci](https://www.linkedin.com/in/anthony-colucci-710659112/)
- [Tanya Tandon](https://www.linkedin.com/in/tanya-tandon/)
- [Saurabh Annadate](https://www.linkedin.com/in/saurabhannadate93/)

## LINKS
- [Dataset](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)
