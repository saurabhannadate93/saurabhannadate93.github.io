---
title: "ProScanner - Handwritten Text Recognition using Neural Networks"
last_modified_at: 2019-06-15T17:00:00-00:00
categories:
  - data-science
tags:
  - Deep Learning
  - CNN
  - RNN
  - LSTM
  - CTC
  - Max Pooling
  - Neural Networks 
classes:
    - wide
header:
  teaser: "assets/images/ProScanner/teaser.jpeg"
---

<style>
figcaption {
  text-align: center;
}

</style>

This blog documents my first deep learning project in which we developed a neural network model using Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) for Optical Character Recognition to convert handwritten text images into digital text.

<figure style="width: 800px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/ProScanner/header.png" alt="">
  <figcaption class="align-center">(Image for illustration purposes only)
</figcaption>
</figure>

## INTRODUCTION
The goal of this project is to build a Handwritten Text Recognition (HTR) model to recognize the text in handwritten text-line images and transate them into the corresponding digital format. This problem is relevant as it finds potential applications in several industries like Legal (digitation of case documents), Banking (Automatic reading of cheques, deposit slips etc.), HealthCare (reading prescriptions, handwritten medical records, insurance forms, general health forms etc.), education, government agencies etc. [1] Digitation of paper documents make them easily searchable, editable, accessible and efficient to store. 

The dataset that was used for building the model was the IAM Handwriting database. The dataset is described in more detail in the Dataset section. An existing implementation of the HTR model used to recognize individual words from this dataset was taken as a starting point. We took this architecture as a baseline and increased this threshold to 200 characters, therefore greatly increase utility in recognizing full sentences.

A google survey was administered to calculate the human correct recognition accuracy rate. A total of 10 random words and 10 lines were used from the model training data to gauge this. For a total of 42 responses, the overall character error  rate was 11.7% (i.e. taking into account all the characters in the 10 words and 10 lines correctly recognized by the 42 responders). For our final model, we got a character error rate of 10.5% for the validation set which is similar to the human character error rate determined.

Although our model was performing satisfactorily, there still are several potential improvements that can be experimented with in order to make the model more accurate and robust. These potential improvements are discussed in the Potential Next Steps Section.

## DATASET
The IAM Handwriting Database [2] [3] was used for training the model. This dataset was first published at the ICDAR in 1999. The database contains forms of unconstrained handwritten text, which were scanned at a resolution of 300dpi and saved as PNG images with 256 gray levels. 657 writers contributed handwriting samples to this database. The dataset contains 1,539 pages of scanned text, 5,685 isolated and labeled sentences, 13,353 isolated and labeled text lines and 115,320 isolated and labeled word. Our primary analysis. Our primary focus of interest was the ~13k labeled text lines which were used for training the model. The dataset contained around 80 distinct characters (lowercase, uppercase, digits, symbols).

The following figure depicts examples of images present in the training data:

<figure style="width: 800px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/house_worth/Fig1.PNG" alt="">
  <figcaption class="align-center">Figure 1: Training Examples
</figcaption>
</figure>

## MODEL TRAINING AND VALIDATION

### MODEL TRAINING CONSTRUCT
The dataset of ~13k observations was divided into 95% training Set (~12.4 observations) and 5% validation set (~600 observations). All 12.4k observations were trained in all epochs and each epoch was divided into 50 batches per epoch. The loss function which was minimized was the tensoflow ctc loss `tf.nn.ctc_loss`. This loss function was used to update the parameters, however we wanted to have a more consumable and intuitive loss function to understand the model performance. In order to do so, we used the *character error rate* (Avg. % of characters wrongly predicted) and the *word error rate* to measure model performance. Early stopping was used to prevent overfitting the model. If the model character error rate did not improve for five consecutive epochs, the model training was stopped and the latest model snapshot was taken as the final model object for that iteration.

## MODEL ARCHITECTURE

Our model architecture involves a model consisting of  7 CNN layers, 2 RNN (LSTM) layers and the CTC loss and a decoding layer.

### CONVOLUTIONAL NEURAL NETWORK (CNN) 
CNN layers extract relevant features from the input image fed during training. The model architecture contains five convolution blocks. The first two blocks consist of two convolution layers with a filter kernel of size 5X5 and RELU activation followed by a pooling layer. The next three blocks consist of one convolution layer with a filter kernel of size 3X3 and RELU activation followed by a pooling layer. While the image height is downsized by 2 in each block, feature maps (channels) are added, so that the output feature map (or sequence) has a size of 200×256.


### RECURRENT NEURAL NETORK (RNN)
Bi-directional Long Short-Term Memory (LSTM) implementation of RNNs is used, as it is able to propagate information through longer distances and provides more robust training characteristics than vanilla RNN. The feature sequence contains 256 features per time-step, the LSTM propagates relevant information through this sequence. The RNN output sequence is mapped to a matrix of size 200×81. The IAM dataset consists of 80 different characters, further one additional character is needed for the CTC operation (CTC blank label), therefore there are 81 entries for each of the 200 time-steps. 

### CONNECTIONIST TEMPORAL CLASSIFICATION (CTC)
While training the Neural Network, the CTC is given the RNN layer output matrix and the ground truth text and it computes the loss value. While inferring, the CTC is only given the matrix and it decodes it into the final text employing Word Beam Search. Both the ground truth text and the recognized text can be at most 200 characters long.


<figure style="width: 800px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/house_worth/Fig2.PNG" alt="">
  <figcaption class="align-center">Figure 2: Model Architecture
</figcaption>
</figure>

#### MODEL EXPERIMENTS







## RESULTS
Our final model performed significantly better to give us a character error rate of 10.5% on the test set.

<figure style="width: 800px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/house_worth/Fig3.PNG" alt="">
  <figcaption class="align-center">Figure 3: Final Character Error Rate
</figcaption>
</figure>


## POTENTIAL NEXT STEPS

## TEAM MEMBERS
1. [Arpan Venugopan](https://www.linkedin.com/in/arpan-venugopal-25312b44/)
2. [Shreyas Sabnis](https://www.linkedin.com/in/shreyassabnis/)
3. [Anjali Verma](https://www.linkedin.com/in/anjaliverma2896/)
4. [Saurabh Annadate](https://www.linkedin.com/in/saurabhannadate93/)

## LINKS
1. [Github](https://github.com/saurabhannadate93/ProScanner)

## REFERENCES
1. [Applications of OCR](http://www.cvisiontech.com/reference/general-information/ocr-applications.html)
2. [IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
3. [IAM Handwriting Database Research Publication](https://link.springer.com/article/10.1007/s100320200071)
2. [Simple HTR for handwritten word prediction](https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5)
3. [Visualizing deep neural network decisions : Prediction Difference Analysis](https://arxiv.org/pdf/1702.04595.pdf)