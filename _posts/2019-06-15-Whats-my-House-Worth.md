---
title: "What's my house worth?"
last_modified_at: 2019-06-15T17:00:00-00:00
categories:
  - data-science
tags:
  - Full Stack App
  - Reproducibility
  - Unit Test
  - AWS
  - Makefile
  - Agile
  - machine learning 
classes:
    - wide
header:
  teaser: "assets/images/house_worth/teaser.jpg"
---

<style>
figcaption {
  text-align: center;
}

</style>

This blog post documents my first foray into developing a full stack analytical pipeline to administer a machine learning solution including using AWS tools such as EC2, S3 and RDS for backend infrastructure and Flask for front end UI. Furthermore, through this project, I got exposure to several good software engineering practices like testing, modularility, reproducibility, logging, managing dependencies, versioning and agile software development paradigm.

<figure style="width: 800px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/house_worth/house_sale.PNG" alt="">
  <figcaption class="align-center">Image for illustration purposes only
</figcaption>
</figure>

## PROJECT CHARTER

### VISION

Real estate agencies require accurate estimation of the price of a property to decide whether it is undervalued or not before making an investment decision. Individual home buyers also need an objective estimate of a home before buying. House pricing decisions are often subjective and can lead to bad investment decisions. The vision is to develop a platform which would help estimate the price of a property based on certain property characteristics to help drive investment decisions, increase profits and reduce costs.

### MISSION

The mission of this project is to build an app which would help accurately predict the price of a property based on certain characteristics like property type, no. of floors, age etc. which can be deployed as a website as well as an Android/iOS app.

### SUCCESS CRITERION

**Modeling ACCURACY**: The model is successful if the modeling accuracy (R-square evaluation metric) exceeds 60%

**BUSINESS OUTCOME**: A Key Performance Indicator of the success of the app would be continual increase in it's adoption to drive business decisions by the various Real Estate agencies and individual customers. This would be a good indicator of the model's accuracy performance as well. The intention is to deploy the app at a particular location, and based on the performance expand to other areas.

## PROJECT PLAN

**THEME: Develop and deploy a platform that helps estimate the valuation of a property based on certain characteristics**

1. EPIC 1: Model Building and Optimization
  - Story 1 : Data Visualization
  - Story 2 : Data Cleaning and missing value imputation
  - Story 3 : Feature Generation
  - Story 4 : Testing different model architectures and parameter tuning
  - Story 5 : Model performance tests to check the model run times

2. EPIC 2: Model Deployment Pipeline Development
  - Story 1 : Environment Setup : requirement.txt files
  - Story 2 : Set up S3 instance
  - Story 3 : Initialize RDS database
  - Story 4 : Deploy model using Flask
  - Story 5 : Development of unit tests and integrated tests
  - Story 6 : Setup usage logs
  - Story 7 : Solution reproducibility tests

3. EPIC 3: User Interface Development
  - Story 1 : Develop a basic form to input data and output results
  - Story 2 : Add styling/colors to make the interface more visually appealing

## DATA DETAILS AND EXPLORATORY DATA ANALYSIS

### DATA DETAILS

The popular Ames housing dataset was used for this analysis (source: [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview)). The dataset contains details of 1,460 properties from Ames, IA region and their sale prices. House attributes included 79 variables including zoning characteristics, neighbourhood details, quality scores, utilities, build year, whether remodeled etc. Looking at data completeness, I observed that there were a few variables that have a very high percentage of missing values. A quick look at the data dictionary revealed that this data was structurally missing, and hence was imputed appropriately. Eg. 93.7% of the records had missing values for the field **Alley**. This implied that 93.7% of all the houses did not have alley access. Similarly, 5.54% of houses in our dataset did not have a garage. All these values were imputed by either **0** or **None** wherever appropriate.

### EXPORATORY DATA ANALYSIS

The house prices in the dataset ranged from $35,000 to $750,000. A quick look at the target distribution (Figure 1) reveals that majority of houses had prices between $80,000 and $400,000. The target distribution was fairly normally distributed with a slight right skew.

<figure style="width: 600px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/house_worth/Fig1.png" alt="">
  <figcaption class="align-center">Figure 1: Distribution of target
</figcaption>
</figure>

Next I looked at scatterplots to understand the relationships between the target and a few numerical variables present in the dataset. Figure 2 shows the scatterplots of the target with GrLivArea (Total above ground square footage) and TotalBsmtSF (Total Basement square footage). Both the plots show that the SalePrice is positively related to both the variables.

<figure style="width: 600px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/house_worth/Fig2_1.png" alt="">
</figure>


<figure style="width: 600px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/house_worth/Fig2_2.png" alt="">
  <figcaption class="align-center">Figure 2: Scatterplots showing relationships between the house sale price and total above ground square footage (above) and total basement square footage (below)
</figcaption>
</figure>

Further, I was interested in understanding whether the time of build of a house impated its sale price. The below boxplots illustrate the relationships between the variables YearBuilt and MSSubClass. Looking at the plots, we can see that the year of build did not necessarily have any impact on the price of the house. Hwowever, the variable MSSubClass revealed a different story. It can be seen that houses built after 1946 had on average higher prices that older built houses.


<figure style="width: 800px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/house_worth/Fig3_1.png" alt="">
</figure>


<figure style="width: 800px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/house_worth/Fig3_2.PNG" alt="">
  <figcaption class="align-center">Figure 3: Boxplots showing relationships between the house sale price and year of build (above) and zoning sub class (below)
</figcaption>
</figure>

Another hypothesis was that the neighbourhood characterstics could potentially impact the sale price. The below boxplots illustrate the relationships between the variables MSZoning and Neigbourhood. From both the plots, we can clearly see that certain neighbourhoods or type of neighbourhoods have higher sale prices as compared to others.

<figure style="width: 800px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/house_worth/Fig4_1.PNG" alt="">
</figure>

<figure style="width: 800px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/house_worth/Fig4_2.PNG" alt="">
  <figcaption class="align-center">Figure 4: Boxplots showing relationships between the house sale price and MS Zoning (above) and neighbourhood (below)
</figcaption>
</figure>

Looking at the correlations of our response variable SalePrice with all the variables (Figure 5), we can see that the highest correlations are with OverallQual, GrLivArea (Above ground living area), TotalBsmtSF (Total Basement Area), 1stFlrSF (First floor area), FullBath (Full bathrooms above grade), GarageCars and GarageArea. Furthermore, the following set of response variables have high correlations between them:

- TotalBSMTSF and 1stFlrSR
- GarageCars and GarageArea
- GarageYrBuilt and YearBuilt
- TotRmsAbvGrd and GrLivArea

Most of the above combinations make sense. E.g. GarageCars and GarageArea are highly correlated. This is intuitive as the number of cars that a garage can accomodate will be a function of the total area available. Including both variables amongst the above sets in our model may lead to multi-dimensionality problems while modeling.

<figure style="width: 800px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/house_worth/Fig5.png" alt="">
  <figcaption class="align-center">Figure 5: Correlation Heatmap of the variables
</figcaption>
</figure>

## MODELING

### DATA CLEANING & FEATURE GENERATION
The target was to predict the **SalePrice** using the other variables. The following variables were dropped from the training data: **YearRemodAdd**, **MiscVal**, **MoSold**, **YrSold**, **SaleType** and **SaleCondition**. The categorical variables were converted to one-hot encoded dummy variables. A new binary variable called **RemodelledFlag** was constructed depending on whether the house had undergone remodeling or not. 

### MODEL BUILDING
For modeling, the sklearn implementation of the *Random Forest Regressor* was used. For the first iteration, all the variables were considered for modeling. *Grid search hyperparameter optimization* was used to identify the best combination of hyperparameters (**max_depth** and **max_features**) optimizing for the OOB (out-of-bag) R-square score. The number of trees were fixed at 1000. The best performing model had max_depth as 22 and max_features as 210 and an OOB R-square of 85.95%. The feature importance plot was evaluated to identify the most impactful variables. It was observed that except a few, most of the variables had very low to no contribution to the model. The model was refit using only the features having feature importances > 0.005. The feature **OverallQual** although being the most impactful was dropped from the model as quality scores were subjective and not an intrinsic characteristic of a house.

The model was retrained using the reduced list of features. The final model contained the following 10 variables:
- GrLivArea
- GarageCars
- TotalBsmtSF
- YearBuilt
- 1stFlrSF
- GarageArea
- FullBath
- LotArea
- TotRmsAbvGrd
- Fireplaces

Since the number of variables were reduced, hyperparameter optimization was again performed to derive the best performing hyperparameters. The best performing model had max_depth = 16 and depth = 6 with n_estimators fixed at 1000. It had a OOB R-square score of 82%. Since this satisfied our modeling success criteron, this was finalized as the final model object. The model object was saved as a pickle file to be integrated into the full pipeline. Figure 6 highlights the feature importances for the final model.

<figure style="width: 500px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/house_worth/Fig6.png" alt="">
  <figcaption class="align-center">Figure 6: Feature Importances for the final Random Forest model
</figcaption>
</figure>

## MODELING PIPELINE

The app has been configured to run in two different modes: **Local** and **AWS** depending on your infrastructure requirements. Please refer to the project README for full set of instructions of how to setup the app.

The following figure depicts the full modeling pipeline:

<figure style="width: 600px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/house_worth/Fig7.PNG" alt="">
  <figcaption class="align-center">Figure 7: Modeling Pipeline
</figcaption>
</figure>

The raw data for this project has been downloaded and uploaded to an open S3 bucket. Depending on the mode, your compute engine can be your local server for mode = 'Local' or an EC2 instance for mode = 'AWS'. The compute engine will fetch the data from the S3 bucket, clean the data, generate features, train and evaluate model and launch the flask app. The model parameters are defined in `config/config.yml`. The app usage information will be logged in a sqlite database (mode = 'Local') or a RDS database (mode = 'AWS'). The usage information can be analyzed for tracking and analyzing app adoption.

An user interface was developed using HTML and CSS to provide a user friendly interface to interact with the model API. The following figure depicts the UI:

<figure style="width: 800px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/house_worth/Fig8_1.PNG" alt="">
</figure>

<figure style="width: 800px" class="align-center">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/house_worth/Fig8_2.PNG" alt="">
  <figcaption class="align-center">Figure 8: Flask app UI home page (above) and form page (below)
</figcaption>
</figure>

## APP DEVELOPMENT ASPECTS AND LEARNINGS

### MODULARITY
Modularity refers to the extent to which a software/Web application may be divided into smaller modules. Every element of the modeling pipeline was controlled by its own submodules within the app. This allowed me to develop and test individual components independently which made debugging and scope of further development easier.

### TESTING
Unit Tests were built in to test individual functions. All tests reside in the `tests/` folder. Unit tests help us understand whether the functions are behaving as desired and help prevent untowardly bugs. It is important to have testing built in specifically in production systems to ensure that the system is behaving as desired.

### LOGGING
The default python `logging` package was used for logging purposes. All logging configurations are controlled via the`config/logging_local.conf` file. The different logging modes (`info`, `debug`, `error`) have been extensively used during development and deployment to debug and keep track of progress. Having descriptive logging messages throughout the code helped me understand what my code was doing and identify bugs, changes in functionality, input data quality, and more.

### REPRODUCEABILITY
Machine Learning solutions need to be reproduceable. This means that anyone having access to the code should be able to replicate the model performance metrics that have been reported for the model. This is important to validate the performance and gain trust and buy-in from the model consumers. In order to ensure that anyone can replicate the modeling pipeline, a Makefile was built to execute the entire pipeline. A Makefile is a file containing a set of directives used by a make build automation tool to generate a target/goal.

## LINKS
1. [Project Repo](https://github.com/saurabhannadate93/Whats-my-House-worth)

## REFERENCES
1. [Exploratory Data Analysis](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python)
