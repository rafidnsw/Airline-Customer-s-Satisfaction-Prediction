# Airline-Customer-s-Satisfaction-Prediction
Final Project of my data science course in Rakamin Academy. The project regarding creation of machine learning prediction model through supervised learning: classification

Data used in this project can be categorized into two category, `personal information` and `ratings`. The project consists of several methodically steps in creating machine learning model:
### Problem Statement
In this first section, we bring up the problem within the company that the data given from. There are 3 points we'd like to emphasize, the objectives, metrics & goals in this project. The details of those 3 points as follow:
* Objectives:
  - Predict the satisfaction level for future customers
  - Determine which features that impactful towards satisfaction level
  - Create an accurate model for the given case
* Metrics:
  - Satisfaction level
  - Average Rating
* Goals:
  - Find features that strongly affect the customer's satisfaction
  - Significant increase on satisfaction level
  - Increased average rating by 0.1
### Data Pre-processing
In this first technical step, we try to define the data and gain a brief summary about our dataset. 

![Dataset Define](https://user-images.githubusercontent.com/85340491/222137058-946641d0-1534-44f3-81e2-4def6afcec30.png)

We found out that there were 14 rating category which scaled from 1 - 5. Based on the summary, we discover that the current average rating for the airline was 3.3 out of 5. Beside that, we also find out from all of the customers taken the survey, **55%** of them **satisfied** with the current airline's services, while **45%** saying they were **dissatisfied**.

On the technical side, in parallel with creating brief summary about our dataset, we also managed to find anomalies in data such as null or duplicated value. This is one of the basic step in the methodology to get rid of those anomalies before we submit our data toward the machine learning model.

![Anomalies Data](https://user-images.githubusercontent.com/85340491/222143464-e416a920-c496-45e8-8a73-572ed20a0a16.png)

From several features available in our dataset, we found that one of the feature called `Arrival Delay in Minutes` has 393 rows missing and there are no duplicated data found. Since we have hundreds of thousand data, we decided to drop out the missing rows from the table, and proceed with the updated dataset.

### Exploratory Data Analysis
In this EDA section we try to find insights that may be useful for our initial purpose. We also visualize most of the features in the data and look for anomalies & supporting facts. We also try to point out outliers in our data, and find the best method to eliminate the outliers without reducing information from the data as optimum as we can.
### Modelling
After done with all the processing and EDA, we now have our data prepared to enter the modelling stage. In this stage we model the prediction using several classification algorithms such as LogisticRegression, DecisionTree, RandomForest & XGBoost. Metrics evaluation that we use is F1-score since we want to truly see our customer that actually satisfied or dissatisfied from our current services.
### Business Recommendation
From all the steps we took in this project, the final output is to give recommendation based on our findings in this project. There are several things that we recommend, mainly about upgrading the services in the airline. In this section we also tried to simulate the impacts that our recommendation can give to the company.
