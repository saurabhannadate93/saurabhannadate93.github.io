---
title: "Text Classification of Amazon Reviews"
last_modified_at: 2019-11-10T17:00:00-00:00
categories:
  - data-science
tags:
  - Text Classification
  - Logistic Regression
  - SVM
  - fasttext
  - Natural Language Processing
  - Neural Networks
  - CNN 
classes:
    - wide
header:
  teaser: "assets/images/tc/teaser.jpg"
---

<style>
figcaption {
}
  text-align: center;

</style>

In this blog post, I explore architectures like Logistic Regression, Support Vector Machines, Convolutional Neural Networks and fasttext to classify Amazon Reviews as positive or negative for a product.

<figure style="width: 600px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/tc/teaser.png" alt="">
</figure>

## INTRODUCTION
Text classification also known as text tagging or text categorization is the process of categorizing text into organized groups. By using Natural Language Processing (NLP), text classifiers can automatically analyze text and then assign a set of pre-defined tags or categories based on its content.

Unstructured text is everywhere, such as emails, chat conversations, websites, and social media but it’s hard to extract value from this data unless it’s organized in a certain way. Doing so used to be a difficult and expensive process since it required spending time and resources to manually sort the data or creating handcrafted rules that are difficult to maintain. 

Text classification is becoming an increasingly important part of businesses as it allows to easily get insights from data and automate business processes. Some of the most common examples and use cases for automatic text classification include the following:

- **Sentiment Analysis**: the process of understanding if a given text is talking positively or negatively about a given subject (e.g. for brand monitoring purposes).

- **Topic Detection**: the task of identifying the theme or topic of a piece of text (e.g. know if a product review is about Ease of Use, Customer Support, or Pricing when analyzing customer feedback).

- **Language Detection**: the procedure of detecting the language of a given text (e.g. know if an incoming support ticket is written in English or Spanish for automatically routing tickets to the appropriate team).

## DATASET
For this analysis, the freely available Amazon Reviews dataset [1] was used. The dataset contains 3,000,000 amazon reviews and their ratings which vary from 1 (worst) to 5 (best). Since fitting multi-class classification models is computationally intensive, the 5 rating values were converted into a binary label. Ratings 1 and 2 were assigned a label of 0 (negative) and 4 and 5 were assigned a label of 1 (positive).

The text data within each review was tokenized using the python regular expression package `re`. It was observed that there were cases where the reviews were blank (no. of words identified = 0). These reviews were dropped from the analysis as we did not have any training data for these. Furthermore, the training data of 3,000,000 was too big to train through all the algorithms efficiently. Therefore, the data was randomly subsampled to 500,000 reviews. 

## METHODS AND RESULTS

### LOGISTIC REGRESSION

### SUPPORT VECTOR MACHINES

### CONVOLUTIONAL NEURAL NETWORKS

### fasttext

## CONCLUSION

## LINKS
1. [Dataset](https://drive.google.com/drive/folders/ (Filename: **amazon_review_full_csv.ta.gz**)0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)
2. [Dataset Research Paper](https://arxiv.org/pdf/1502.01710.pdf)
3. [Github](https://github.com/saurabhannadate93/Text-Analytics-Amazon-Reviews-Text-Classification)

https://monkeylearn.com/what-is-text-classification/

## REFERENCES
