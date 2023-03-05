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
  
After we have a clear image on the objectives & goals for the project, we may continue to the technical steps taken while processing our data into a machine learning model.

### Data Pre-processing
In this first technical step, we try to define the data and gain a brief summary about our dataset. 

![Dataset Define](https://user-images.githubusercontent.com/85340491/222137058-946641d0-1534-44f3-81e2-4def6afcec30.png)

Our whole dataset numbered in close to 130.000 data where there are 23 features exists excluding 1 target feature which will be our main objective to predict.
We found out that there were 14 rating category which scaled from 1 - 5. Based on the summary, we discover that the current average rating for the airline was 3.3 out of 5. Ultimately, based on the data it is known that the result of customer's satisfaction level, where **55%** of them **satisfied** with the current airline's services, while **45%** saying they were **dissatisfied**.

On the technical side, in parallel with creating brief summary about our dataset, we also managed to find anomalies in data such as null or duplicated value. This is one of the basic step in the methodology to get rid of those anomalies before we submit our data toward the machine learning model.

![Anomalies Data](https://user-images.githubusercontent.com/85340491/222143464-e416a920-c496-45e8-8a73-572ed20a0a16.png)

From several features available in our dataset, we found that one of the feature called `Arrival Delay in Minutes` has 393 rows missing and there are no duplicated data found. Since we have hundred of thousands data, we decided to drop out the missing rows from the table, and proceed with the updated dataset.

There are some features that we engineered for the purpose of creating dataset to be more comprehensible for the model to read. Such as labelling our target feature `satisfaction` values from `satisfied` & `dissatisfied` into a feature called `satisfaction_label` with value of 1 & 0 consecutively. Before submitting to the model we create, we also correct the outliers found in the dataset using IQR method while simultaneously normalizing the data.

![IQR](https://user-images.githubusercontent.com/85340491/222880321-7192ef58-6aa2-4a42-a4b5-7c2280cebee0.png)
_From left to right: Distribution on Raw data, Normalized, IQR method_

### Exploratory Data Analysis
In this exploratory section we are dissecting the data even further to find insights that may be useful for our objectives. One of our favorite method initially is to create a correlation map using Heatmap. This heatmap really helpful to display the correlation in the value between features. The closer the heatmap value to 1 indicate stronger correlation between data, whereas the closer the value to 0 means weaker correlation.

![Heatmap](https://user-images.githubusercontent.com/85340491/222899913-772c787d-80c6-4c0a-a449-5709e679c3a7.png)

From the heatmap we can see some feature correlate really strong to each other such as `Departure Delay in Minutes` & `Arrival Delay in Minutes`, and here is when our business understandment required. As we can probably infer, the departure & arrival time were indeed impactful toward each other. It is a common sense when the departure time got delayed, the arrival time correspondingly will also get so. Hence, when there are cases like this, we can just drop one of the feature from our dataset since **it will only add bias** to our model later. Eventually we simplified the heatmap as below.

![Heatmap2](https://user-images.githubusercontent.com/85340491/222900888-c28a1178-7bb5-4f52-b975-2c0e559b5916.png)

Some other insights that we can derive from this heatmap was to select features that probably more impactful in the dataset. We defined there were 6 features that indicate stronger correlation towards our target value `satisfaction_label`. Those 6 features are `Seat comfort`, `Inflight entertainment`, `Online support`, `Ease of Online booking`, `On-board service`, and `Online boarding`.

Aside from getting insights on the rating, we also explored the personal information data. We managed to group customer satisfaction based on some segments. From the charts below we can conclude that firstly, comparing with the male customer, female customer are more satisfied toward the current airline services. Based on the customer type it is quite obvious that the loyal customer tend to have higher satisfaction than the disloyal ones. Next on, in this dataset there are 2 kind of travel type which are 'Personal Travel' & 'Business Travel'. From the chart it can be inferred that the number of customer with personal travel kind of type numbered lower compare to business travel in which also provide lower satisfaction than those who have business travel kind of type. This conclusion also has relevancy to the last chart where business traveller more likely to travel with the 'Business' class airline, thus the satisfaction level of the 'Business' class was higher compared to the 'Eco' and 'Eco plus' class.

![satisfaction segment](https://user-images.githubusercontent.com/85340491/222901467-8a6faf67-2786-401b-a3ec-8b38e547db2f.png)

Lastly, from our analysis we found that there are several services that giving a significant value of 'satisfied' customer in their rating value of 5. Those top 3 rating that provide the higher number of satisfied customer in their rating value of 5 were `Inflight entertainment`, `Seat comfort`, & `Ease of Online booking`. To make this more comprehensive, we provide the _satisfaction probability chart_ as below.

![Probability chart](https://user-images.githubusercontent.com/85340491/222903657-a93ef4fc-2a14-4465-bad9-b50118814ad1.png)

To put it simply, this chart tell us the chances (in percentage) the customer will submit 'satisfied' value in their satisfaction survey. As we can see, this 3 rating category have the most significant improvement from value rating of 3 to value rating of 5. Lets take the `Seat comfort` rating for an example, from the chart we can infer that the probability of customer with value rating of 3 saying they are 'satisfied' was around 45%. However, if we managed to increase their rating value to 5, we can see an almost 100% probability of saying 'satisfied'. This chart can be our guidance to take our way forward of the service improvement for our business recommendation.

### Modelling
After finish all the work on the data pre-processing and gain in-depth knowledge and insights regarding the data, we continue to build our classification model. On this project, we tested several classification algorithm. Those algorithms are:
  * Logistic Regression
  * Decision Tree
  * K-Nearest Neighbors
  * RandomForest
  * XGBoost

Since our objective in this project was about improving the rating and services of the airline, we would like to comprehend most of the positive and negative outcomes from both our 'satisfied' and 'dissatisfied' customer. Therefore, we decided to choose the F1-score for our evaluation metric. This choice was taken since F1-score pretty much combine the intuition of precision and recall score in one formula.

![Model Selection](https://user-images.githubusercontent.com/85340491/222952518-7e31bf8f-0789-4bbf-9d20-fdb6f42e3da3.png)

Based on our model results, we got quite same score between RandomForest and XGBoost algorithm. In this case, we prefer to proceed with RandomForest over XGBoost because the basics of the algorithm itself. XGBoost provide us result with a sequential type model thus it is more prone to noises than RandomForest.

After finishing with our model, we try to extract feature importance from our model. We discover the top 3 features that most important in our model are `Inflight entertainment`, `Seat comfort`, & `Ease of Online booking`. These 3 features are in-line with our analysis back on the EDA section. Therefore, we are confident to provide business recommendation based on this result.

![Feature importance](https://user-images.githubusercontent.com/85340491/222953339-67685d06-c218-4b27-962f-8a82622efb7b.png)

### Business Recommendation
Based on our predictive model, we highly recommend for the business team to focus giving improvement on these 3 services that impactful towards overall rating and satisfaction survey:
  1. Seat comfort: Renewing the airplane seat, replacing them with more ergonomic bench model and paying attention to body posture when sitting
  2. Inflight entertainment: Support output like headset and monitor, periodically update the options existing entertainment such as movies, music or other media services
  3. Ease of Online Booking: Create an easy-to-understand interface so that users don't feel bothered and having more efficient time when making a booking
To build up more confidence on our recommendation, we also create an impact simulation as shown below:

![Impact model](https://user-images.githubusercontent.com/85340491/222955088-fb9c2bda-2074-4db3-83af-b27d1212bd94.png)

Here we can see there are 3 cases we try to simulate and also on the table shown, we can see there are 6 category services take as samples where in each cases we improved 3 of 6 category and assume the customer rating narrowed to the value of 4 and 5 only. The first case was the general implementation where the business recommendation applied to **every** `Class` and it is assumed we succesfully increase the rating of `Inflight entertainment`, `Seat comfort`, and `Ease of Online booking` to the value rating of 4 and 5. Next one, is where we implement the improvement on random services such as `Departure/Arrival time convenient`, `Gate location`, & `Inflight wifi service` and succesfully provide value rating of 4 and 5 on those services. The last case was to implement the business recommendation only on 'Eco' & 'Eco Plus' `Class`.

With the assumption of airline customer number in Indonesia was 10.000 per day, we can achieve different results within 1 week. From the chart shown the best outcome is when we implement the business recommendation in Eco-focused segment where 15.5% satisfaction level increasement can be expected. The general implementation also giving significant level of increase in satisfaction however it will cost the company more since all type of classes being improved. For the random improvement, it is shown there are no progressing on the satisfaction level increase, even worse the satisfaction level shows a decrease of 0.3% in value.
