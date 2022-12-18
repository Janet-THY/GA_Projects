# Capstone Project - E-Commerce Analysis and Demand Forecasting

## Executive Summary
The key idea of this project is to use data science to help E-Commerce business owners (so-called `seller` in this dataset) to forecast the demand for their products or services. There are three goals for this project. Firstly, it is to better understand the data within the e-commerce site especially the sales order demand from the perspective of time series. Secondly, it is to design and build a time series forecasting model for sellers to forecast the sales order demand. Lastly, it is to create a forecast application to allow sellers to predict and visualize the sales demand. 

The data used was sourced from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce),generously provided by Olist, the largest department store in Brazilian marketplaces. With purchase date information provided, this enables us to predict future sales demand using time-series. The performance metrics used to evaluate the models are RMSE and SMAPE.  

The exploratory analysis shows that E-Commece industry has upward trajectory trend in Brazil. Most orders were purchased on Monday and least on Saturday. Customers tend to purchase during 4-10pm of the day. Top most selling product catergories and their time-trend similar product categories were identified. The data is pre-processed on the purchases count and ran through two stationary tests - ADF and KPSS to check for stationarity. The best forecasting model, Exponential Smoothing forecasting model on Seasonal-Treand-based-on-Loess(STL) transformed data, showed RMSE of 67.18 and SMAPE of 21.36%. The forecasting webapp was built on Streamlit.


## Introduction & Problem Statement
At the era of promoting sustainability in business, organizations face this challenge along with ways to maximizing profit and reducing cost (e.g. investment cost, packaging cost, shipping cost, warehouse storage cost). The least of the situation a business wants is to overly produce a product or to turn down potential customers due to shortages. 

With the aim to enhance the services/ experience for sellers and grow sellers' group, I wish to leverage on this project:
- To better understand the databse within the e-commerce site, 
    - overall order demand over time
    - e-commerce's impact on economy
    - customer base by location (does this matter for e-commerce?)
    - which product is popular & in high demand
    - top selling product categories 
- To build a time series forecasting model for sellers to forecast the sales order demand up to 2 months, with the least RMSE and target SMAPE of 25%.


## Goals/ Key Questions
1. Sales orders trend analysis
    - How is the sales trend presented in the dataset? Can we describe the sceanrio? What are some features that affect this trend over time?
2. Forecasting model
    - Test and build model to  forecast sales demand for next 2 months.
3. Forecast application
    - Build a forecast application to visualize time-series data.


## Background 
[*E-commerce*](https://www.toppr.com/guides/business-environment/emerging-trends-in-business/electronic-commerce/) is a popular term for electronic commerce or even internet commerce. As the name suggested, it's the meeting of buyers and sellers on the internet. This involves the transaction of goods and services, the transfer of funds and the exchange of data. These business transactions can be done in four ways: Business to Business (B2B), Business to Customer (B2C), Customer to Customer (C2C), Customer to Business (C2B). Online stores like Amazon, Flipkart, Shopify, Myntra, Ebay, Quikr, Olx are examples of E-commerce websites.

Over here, the seller exchanges data in the form of pistures, text, address for delievry etc. and then buyer make the payment. As of now, e-commerce is one of the fastest growing industries in the global economy. As per one estimate, it grows nearly 23% every year. And it's projected to be a [*$27 trillion*](https://www.insiderintelligence.com/insights/ecommerce-industry-statistics/) industry by the end of this decade or by 2020. 

E-commerce industry has steadily gained popularity. In today's world, E-commerce industry is facing [*potential challenges*]() & [*limitations*](https://www.notifyvisitors.com/blog/limitations-of-ecommerce-business/). Just to name a few:
- Huge technological cost 
- Security
- Employee cost
- Investment/ Advertising cost
- Shipping cost 
- Packaging cost
- Warehouse storage cost 
- Marketing cost
- Complicated eCommerce policies
- Sales flow
- Customer Relationship Management(CRM) maintenance 

[*Demand and sales forecasting*](https://mobidev.biz/blog/machine-learning-methods-demand-forecasting-retail) are of paramount importance in retail. Without this tool, companies encounter disruption of the inventory balance, through ordering too much or not enough products for a certain period of time. In the case of surplus, a company is forced to offer discounts to sell products. Otherwise, it may face inventory issues. A shortage, in turn, results in lost profits. However, these problems can be solved by applying demand and sales forecasting to increase the return on inventory and determine the intention of future consumers to buy a specific product at a specific price.   


## Dataset
The data used was sourced from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce),generously provided by Olist, the largest department store in Brazilian marketplaces. 


## Table of Contents
1. [Introduction and Data Joining]('01_Introduction_Data_Joining.ipynb')
2. [Exploratory Data Analysis (EDA), Feature Engineering and Preprocessing]('02_EDA_Feature_Engineering.ipynb')
3. [Demand Forecasting Modelling]('03_Demand_Forecasting_Modelling.ipynb')


## Summary of EDA
Datasets are merged sequentially and information is extracted and aggregated to meet the purpose of the visualisation.

Summary of observation from the extensive EDA process:
- From the statistics graph, the status with highest count of the orders is *delivered* (97%). Whileas the remaining orders is spread among the other statuses.
- Concept of `ETL` is applied, such as to extract time attributes from timestamp columns, transform to different datetime columns (year, month, day, day of the week and hour), analyze the e-commerce scenario using these attributes. 
- Customers tend to shop online on weekdays from 10am to 5pm. There are sudden peaks on Monday-Wednesday night around 8-9pm and on Sunday evening 6-8pm. Online shopping count is relatively low on Saturday.
- The delivery heatmap shows that Weekday 2-10pm are the busiest time for delivery operators. Morning periods is generally the time with less deliveries. 
- E-Commerce in Brazil has positive growing trend along the time. There is some seasonality with peaks at certain months, in general we clearly see that customers are more keen to buy things online than before.
- Monday and Tuesday are the preferred days for Brazilian customers and more puchases are made in the afternoon.
- There is sharp decrease after August 2018, it may due to the noise on data or incomplete data. For comparison purposes, we focus on orders between January and August for both 2017 and 2018. 
- Between January and August, total orders registered in 2018 has 142% more than in 2017.
- An [API](https://servicodados.ibge.gov.br/api/docs/localidades?versao=1) (brazilian government) was used to return the region of each customer_state, merge this data with orders_items_df to remove any possible outliers.
- There are five main regions where the purchases demand are coming from, top region being the southeast. São Paulo is the most populous city in Brazil, it has the highest number of sales among all. Follows by city Rio de Janeiro (capital of Brazil 1763-1960). While the current capital of Brazil, Brasilia, is the fourth capital with most orders.
- We analyzed the money movement of e-commerce. Highest amount sales sold in this dataset was R$ 1010.3K on November 2017, which was possibly the result of Black Friday (Nov 24). Total sales amount of R$7.39Mil generated between January and Auguest 2018. Freight for most months in 2018 are higher than in 2017.
- It's very interesting to see some states with a high total amount sold but low price per order. For example, in SP (São Paulo) state, it has the most value of e-commerce (5,188,099 sold) but it is also where customers pay less per order (110.00 per order).
- Here we can get insights about the customers states with highest mean freight value. For example, customers in Roraima (RR), Paraíba (PB), Rondônia (RO) and Acre (AC) normally pays more than anyone on freights.
- The mean value of freight paid for online shopping is R$19.99. It generally takes average of 8 working days for online shoppng delivery. While the avearge difference between delivery and estimated date is -8 working days, which means delivery is faster than estimated. 
- From the line chart, it shows that majority of brazilian e-commerce customers made payments by credit card. Since May 2018 it's possible to see a slight decrease on this type of payment. On the other hand, payments made by debit card is showing a growing trend since May 2018, wich is a good opportunity for investors to improve services for payments like this.
- On the bar chart, it reflects that brazilian customers prefer to pay the orders: mostly of them pay all in one installment and it's worth to point out that payment in 10 installments comes after the quantity of payments done in 4 installments.
- 89638 of customers purchased once (single order_id) while another group of customers (7516) are repeating customers.
- Both supply by sellers and demand from the customers exist in Olist. However, we have no information on when a product was first launched for purchase. This is relevant beacuse it would allow us to analyze the relationship between supply (product availability) and demand (purchase of products). It seems that the product information we have is only related to purchased items which makes it fully derived and dependent from the demand side.
- Product price and freight value exhibits same distribution pattern. As expected, both price and freight value plots are right skewed histogram - most of the times, people buy cheaply priced goods on Olist.
- There are some sales in October 2016, but almost nothing in the next two months.
- There is a spike in products price in July 2018, but not in freight.
- The unique counts of orders, products, sellers and customers have similar cumulative distribution function. All of them skew to the right and in the case of daily unique orders, it reached 90% at around 300 orders but reached 100% at over thousand orders.
- The orders between October 2016 and January 2017 are very little to almost zero, we'll be removing this interval from the dataset for our modelling.
- It looks like there is steady gradual upwards trend in the first 3 quarters of 2017 follow with huge spike in November and December 2017. From the spike at the end of 2017 onwards, daily orders oscillate much more heavily and it's hard to tell if there is any trend at all. There are two additional huge spikes in the second and third querter of 2018.
- Moving averages was applied to time series plot at different granularity to observe the trends through smoothening. Through these averaged, we can definitely better recognize the steady growth through 2017 but also no growth in the first two quarters of 2018 and then a dip in the middle of the 3rd quarter of 2018. 
- There are 74 categories for which a product was shipped as part of an order any given point in time. The top 10 categories make up 63% of the total products sold during the two years of data.
- The top most selling product category is `bed_bath_table`, followed by `health_beauty` and `sports_leisure`.
- Almost all the top ten categories maintained same monthly share of products sold with some fluctuation, however we observed some odds such as huge drop for furniture_decor products on beginning of 2017 and spike for computer_accessories on March 2018.
- Heatmap was built to spot the trend with time for top ten categories. The top category for the two years of data `bed_bath_table` consistently takes around 8% to 10% of monthly products sold. `health_beauty` has an upwards trend starting in the lower 7% and ending the last months of the series above 10% from total monthly sold products, overtaking bed_bath_table as the top category for these months. The categories `housewares` and `watches_gifts` had an upwards trends for their share of total products sold towards the end of the series, while `sports_leisure` and `furniture_decor` had a downwards trend as time went on.
- In terms of the share from total monthly products sold per city, we can see that as the number of products sold increased (see figue 'Monthly Products Sold'), the share of Sao Paolo increased as well, meaning that most of the growth in products sold came from customers in Sao Paolo. Rio de Jairo has a more or less constant share of between 6% to 8%. All other cities, the long tail, seem to have a mothly share of 2% or less.
- From the heatmap, all cities seem to have a similar percentage change to their cumulative monthly products sold. This means that in overall, their cumulative plots have a similar shape even though the absolutes might be of different magnitudes.
- We conducted seasonlity check with using stationary test. 
- The exponentially weighted average places more weight on days closer to the day for which the average is being calculated. There is in fact a daily seasonality, high number of customers orders are consistently being approved on Tuesday than any other day of the week. It also seems that Wednesday is the second day where most orders are approved.
- Until December 2017, according to the Dickey-Fuller test, the times series is not stationary and it clearly has a trend upwards. The p-value is 98% for this time range. The rest of the time series seems to be stationary, the test confirming the null hypothesis with a very low p-value. Let's plot p-values for the interval ranges of 3 months to see how consistent the above results are.
- Scores are usually good and the bad ones seems to be related to slightly more expensive products (or considerable orders).
- We observed low variation in the score. Majority of the scores are between 3.5 to5. The product category with lowest score is `security_and_services`
- `Fashion_childrens_clothes` has the highest average score of 4.64 among all product categories.
- Scores are usually good (value of 4/5) and the bad (value of 1) ones seems to be related to slightly on more expensive products (or considerable orders).
- Review score increases with shorter delivery time. We observed low variation in the score. Majority of the scores are between 3.5 to5. The product category with lowest score is `security_and_services`
- Some new features being engineered along the EDA processes, such as time attributes, holiday, region.
- Interested to explore category simialrity algorithmn potentially used for recommender system, but this will be future work.


## Modelling
In this section we focus on Univariate time series forecasting methods which capable of only looking at the target variable. This means no other regressors (more variables) can be added into the model.

<u> Objective </u>
* To forecast order counts for the next two months using Time-Series model, with the least forecast error in terms of root_mean_square_error (RMSE) and less than 25% for symmetric_mean_absolute_percentage_error (SMAPE). As according to [white paper](https://www.e2open.com/wp-content/uploads/2019/02/2018_Forecasting_and_Inventory_Benchmark_Study_white_paper_digital.pdf)  in 2018, the most forecastable business have an error of 29%.

<u> Experiment Setup </u>
* Enviroment: 5-fold Time-series cross validation (7-days forecast in the Test Set). Find the best combination between 4 data transformations + 4 time-series models. 

<u> Performance Metric </u>
* Metrics: Model is selected based on the least average RMSE and SMAPE.  
___
A model performance can be tested by evaluating a comparison between the actual values and the predicted values. In this project, two performance metrics, namely the root mean square error (RMSE), symmetric mean absolute percentage error (SMAPE) are used to evaluate the performance of each model.
___

<u> Baseline </u>: The original time-series dataset for each model without data tranformation. 

<u> Experiments Components </u>:
* [4 Data transformations](#trans):
  * Original time-series dataset
  * Time-series Deseasonalized with STL Decomposition
  * Time-series Transformed with Box-cox Tranformation 
  * Time-series with Seasonal Differencing 
* 4 Time-series models:
  * [Time-series regression](#res_reg)
  * [Holt-winter's exponential smoothing](#res_es)
  * [Auto ARIMA](#res_arima)
  * [Long-short term memory/ LSTM](#res_lstm)
* Grid-search hyperparameters space:
  * Holt-winter's exponential smoothing
    * Smoothing Level, $\alpha$: 0, 1, 2, ..., 7, 8, 9
    * Smoothing Trend, $\beta$: 0, 1, 2, ..., 7, 8, 9
    * Smoothing Seasonal, $\gamma$: 0, 1, 2, ..., 7, 8, 9
  * LSTM
    * Adam learning rate: 0.001, 0.01, 0.1, 1, 10

## Model Evaluation 
Within the scope of this project, from the metric evaluation, the best model with least RMSE and SMAPE is Exponential Soothing + STL. 
- Top 3 models with minimal RMSE belongs to Exponential Smoothing family. 
- Data transformation method STL has been efficient in bring down the RMSE in this case. 
- Seasonal differencing data with exponential smoothing performed the worst among all. 

|        Model        |    RMSE   | SMAPE |   Category   | 
|:-------------------:|:--------------:|:-------:|:----------:|
|        Exponential Smoothing + STL  |	67.18  |  21.36  | 	Exponential Smoothing         | 
|  Exponential Smoothing  |	 73.15  |	21.90 |  Exponential Smoothing  |
|	Exponential Smoothing + Boxcox  |	73.59  |	21.74  |	Exponential Smoothing  |
|	LSTM + STL  |	76.73  |	27.51  |	LSTM  |
|	Time-Series Regression + STL  |	77.20  |	27.67  |	Time-Series Regression   |


## Conclusion
By using the best model to forecast the sales for subsequent two months, we expect the daily order count has a constant growth with the weekly seasonal applied after 17 August 2018. The order count shouldn’t be a drop as given from the data. For the forecasted period between 17 August to 17 October 2018, both storesellers and Olist should prepare more resources to meet the surge of order purchase demand.

In conclusion, we want to forecast the subsequent two months’ order count with the best forecasting model. The Exponential Smoothing with STL transformation can be applied on the original data to yield the lowest RMSE.

## Limitations of the modelling approach
1. The selected models are currently based on univariate factor. Multivariate factors can also be considered and explored with the addition of exogenous variables. There are many factors account for forecasting sales demand.   
2. There are many more advanced time series forecasting models are not being tested yet, such as Bidirectional/ Stacked LSTM or hybrid models with Prophet. 
3. The current models include only from 2017-01-01 to 2018-08-17. It will be better if we can observe time series components for another year.
4. Geolocation and correlated factor of this dataset have not being considered yet. 


## Recommendation
**Forecastability of business**
1. To make a business more forecastable is to manage more aspects of the supply chain. Direct store delivery supply chains typically have 10% lower weekly error than warehouse-delivered businesses. The two key reasons for this are direct visibility into consumer demand and control over retailer execution. Companies for which direct store delivery is not a feasible option should consider expanding vendor-managed inventory programs and leveraging store data to sense demand at retailer distribution centers.
**Demand forecasting**
2. To better understand respective market demand, storesellers can start looking at their own product catergory. Then examine each category on a national level to determine what was driving the demand for these products. From there, they can dive into a more granular level, taking in economic factors for certain markets such as employment and cost of living to build predictive models. With this information, the storesellers will able to build predictive models for each of its market areas, which procurement used to stock shelves with the goods consumers were ready and able to purchase.
3. Storesellers can also practise more proactive demand forecasting which does not only make use of historical data but external factors too.
4. As future is uncertain, forecasts must often be revised, and actual results can vary greatly.    


## References 
1. [2018_Forecasting_and_Inventory_Benchmark_Study_white_paper_digital](https://www.e2open.com/wp-content/uploads/2019/02/2018_Forecasting_and_Inventory_Benchmark_Study_white_paper_digital.pdf)
2. [Visualiation Dashboard](https://www.kaggle.com/code/thiagopanini/e-commerce-sentiment-analysis-eda-viz-nlp)
3. [Similarity Algorithmn](https://github.com/LarsTinnefeld/olist_ecom_analysis)
4. [Time Series](https://github.com/juloi/udacity_ds_nanodg_blog_post/tree/master/olist_ecommerce_eda)
