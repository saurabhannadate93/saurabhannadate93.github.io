---
title: "Language Modeling of Gutenberg Corpus"
last_modified_at: 2019-12-10T17:00:00-00:00
categories:
  - data-science
tags:
  - Deep Learning
  - CNN
  - RNN
  - LSTM
  - Natural Language Processing
  - Neural Networks 
classes:
    - wide
header:
  teaser: "assets/images/lm/teaser.jpg"
---

<style>
figcaption {
  text-align: center;
}

</style>

As a part of the final course project for Text Analytics, I decided to build language models using the freely available Gutenberg corpus. This blogs documents the experiments and findings of this project

<figure style="width: 600px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/lm/languages.png" alt="">
</figure>

In this project, I explore three model architectures which include statistical N-gram models and two Recurrent Neural Network architectures for large scale language modelling. A subset of the Gutenberg eBooks corpora was used as the corpus on which the models were fit. Several hyperparameters were tested and the performance metrics as well as the training time for the models were recorded and analyzed. Due to the difference in the architectures, the model performance metrics cannot directly be compared amongst the different architectures.

## INTRODUCTION
Language modeling (LM) is the task of modeling the probability distribution of the sequence of words/characters in a corpus. It plays a key role in several NLP tasks such as speech recognition [1][2] and text summarization [3][4]. Therefore, improving the LM performance is very valuable as it will improve the underlying metrics of the downstream tasks. Language Models also find their application in seq-to-seq models which are used in machine translation [5][6] and video generation [7]. 

There are two primary types of language models - Statistical Models which are the traditional models and Deep Learning models which are more recent. The statistical models use techniques like N-grams, hidden markov models, and certain linguistic rules to learn the probability distribution of words. The Deep Learning Neural network models use neural networks to learn the language structure to predict the next word/character given an input context.

In our work, we train the models on a subset of the Gutenberg eBooks corpora which can be considered as a medium sized dataset for the n-gram models but a very large size dataset for the neural network models. This dataset has not been used for language modelling before. For one of our RNN models, we experiment with a different unseen architecture which models the embedding of the next word rather than the word itself.  

## RELATED WORK
In this section, we discuss previous work relevant to the approaches discussed in the paper.

### LANGUAGE MODELS
The aim of a language model is to learn a probability distribution over sequences of symbols pertaining to a language. A lot of work has been done on both parametric as well as count-based approaches (N-gram models). In the past five years, a lot of progress has also been made to train deep learning neural network models to learn the language distributions with good accuracy, especially Recurrent Neural Networks and its variants which can learn and retain long term dependencies. 

### N-GRAM MODELS
An N-gram model predicts the probability of a given N-gram within any sequence of words in a language. If we have a good N-gram model, we can predict p(w/h) - what is the probability of seeing the word w given a history of previous words h - where the history contains (n-1) words. The way the model achieves this is that it counts how many times the word w appeared in the context h, and normalizes by all observations of h. Although N-gram models are theoretically sound, they have a few limitations [8]:

1. Many histories h are similar, but n-grams assume exact match of h 
2. Practically, n-grams have problems with representing patterns over more than a few words
3. With increasing order of the n-gram model, the number of possible parameters increases exponentially
4. There will be never enough of the training data to estimate parameters of high-order N-gram models

In order to model unseen sequences, N-gram models typically add smoothing which have been quite successful. For instance, Kneser-Ney smoothed 5-gram models have challenged few of the other parametric approaches including neural networks.

### GATED RECURRENT UNITS AND LONG SHORT TERM MEMORY
For our neural network architectures, we have experimented with Gated Recurrent Units and Long Short-Term Memory which are forms of Recurrent Neural Networks which can retain long term dependencies and do not suffer from vanishing gradient problems. This is achieved by internal mechanism called gates [9] which control the flow of information. These gates can learn which information in data is important and which can be thrown away. The architectures that we considered are highlighted in the subsequent 

### word2vec
For one of our neural networks, we train the model on the word embeddings for the input sequence of words and predict the word embedding of the next word. A word2vec model is used to construct both the input word embeddings as well as to derive the next word based on the predicted embedding. word2vec [10] is a popular word embedding technique which learns the distributed representation of words using neural nets. The learnt vectors are able to maintain linear semantic and syntactic relationships. There are two types of models called CBOW (cumulative bag of words) and Skip Gram. For our analysis, we fit a skip-gram model using hierarchical softmax to reduce the word count complexity.

## DATASET
The [Project Gutenberg corpus](http://www.gutenberg.org/) was considered for our analysis. Project Gutenberg is a library of over 60,000 free eBooks. The books in the project repository have been chronologically assigned a serial number which goes from 1 to ~62000. All files are stored as “UTF-8” encoded txt files. We have considered books from serial number 45,000 to 62,000 for our analysis. Once the books are downloaded, the following operations are performed to create the final corpus:

1. Books are filtered for English books
2. Book metadata and the Gutenberg license is removed from all the books’ texts
3. First 500 books’ texts are concatenated
4. The final corpus is cleaned by removing the occurrences of unwanted characters like ‘\n’ and multiple spaces between words
5. The sentences were padded with begin and end of sentence tokens for the N-gram modelling

After creating the final corpus, the text is tokenized using NLTK word_tokenize and sent_tokenize. The final corpus contains 186,928,914 characters and 39,120,715 words.

For the character level and word level neural nets, the training corpus was restricted to the first 1,800,000 units (characters for character level neural net and words for word level neural net) and the validation set was the next 200,000 units. The full corpus was taken for the N-Gram models.


## METHOD
The following three models were fit on the data for language modelling:

### N-GRAM MODELS
We trained four N-Gram models for N = 2, 3, 4 and 5 using NLTK lm module. This basic module does not perform any smoothing and hence outputs the probabilities for unseen sequences as zero. As a result, we were unable to calculate the perplexity metric which is generally used to report performance of language models. 

### CHARACTER LEVEL NEURAL NET
Figure 1 depicts the model architecture used for character level neural net model. The input is a sequence of 128 characters and the model aims to predict the next character. The model consists of a Keras embedding layer which creates embeddings of dimension 128 for the input characters. This is followed by two or more recurrent neural network layers with varying hidden states. The final layer is a softmax dense layer over the entire character set which calculates the probability of the next character. There are 157 unique characters in our training set over which the model calculates the probability. Different model architectures are constructed by varying the number of hidden states of the RNN, number of RNN layers as well as type of RNN unit (LSTM or GRU). The loss that is minimized is categorical cross-entropy. All the models are trained for 5 epochs and the training time and validation cross entropy is recorded.

<figure style="width: 600px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/lm/Fig1.png" alt="">
  <figcaption class="align-center">Figure 1 : Character Level Neural Net model architecture
</figcaption>
</figure>

### WORD LEVEL NEURAL NET
The biggest challenge with constructing a word level neural net is that the final softmax layer needs to predict over the entire vocab which may contain millions of words and hence becomes computationally intensive. Although there are techniques to count this like Noise Contrastive Estimation (NCE) loss, self-normalizing partition functions, or hierarchical softmax, we decided to adopt a slightly different approach. Instead of modelling the probability of the next word, we modelled the word embeddings. Figure 2 shows the model architecture. First, a word2vec model is fit on the entire training corpus. A sequence of 128 words is passed through the word2vec model to get the embeddings which form the input for the neural net model. This is followed by two or more recurrent neural network layers with varying hidden states. The final layer is a tanh dense layer of dimension 300 which predicts the embedding of the next word. The loss which is optimized for is mean square error.  Different model architectures are constructed by varying the number of hidden states of the RNN, number of RNN layers as well as type of RNN unit (LSTM or GRU). All the models are trained for 5 epochs and the training time and validation mse is recorded.

### TRAINING PROCEDURE
Dropout of 0.3 and a l2 penalty of 0.0003 was used in all neural net models. All models were trained in batches of 250 for a total of 5 epochs on a GeForce GTX 1080 8gb dedicated GPU.

## RESULTS

### N-GRAM MODELS
Table 1 shows the Train Time and the Model size for the N-Gram models. As it is seen, the models do not take substantial time to fit, however as the N count increases, the model parameters and hence the model size increases exponentially. As it is noted before, since our model does not incorporate any smoothing, we are unable to report any performance metrics like perplexity.

| Model	| N-Grams	| Train Time	| Model Size |
|----|----|----|----|
| Model 1	| 2	| 8min | 110mb |
| Model 2	| 3	| 14min	| 513mb |
| Model 3	| 4	| 23min	| 1.40gb |
| Model 4	| 5	| 34min	| 2.81gb |










## LINKS
1. [Github](https://github.com/saurabhannadate93/Text-Analytics-Language-Modeling)

## REFERENCES
