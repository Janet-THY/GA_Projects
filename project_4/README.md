# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) West Nile Virus Prediction

## Introduction

The West Nile Virus (WNV) has been a worrying disease for the United States since 1999. The [CDC](https://www.cdc.gov/westnile/index.html) has acknowledged it as the leading cause of mosquito-borne disease feeding on infected birds [(*source*)](https://parasitesandvectors.biomedcentral.com/articles/10.1186/1756-3305-3-19#citeas) in the continental United States. Now, there are still no vaccines to prevent or medications to cure WNV patients -- statistics data from the CDC, West Nile fever (WNF) is a potentially serious illness for humans and approximately 1 in 150 infected people develop a serious illness with symptoms that might last for several weeks. Up to 1/5 of patients have milder symptoms and approximately 4/5 show no symptoms at all [(*source*)](http://www.cdc.gov/westnile/faq/genQuestions.html).

In Illinois, [West Nile virus was first identified in September 2001](https://www.dph.illinois.gov/topics-services/diseases-and-conditions/west-nile-virus) when laboratory tests confirmed its presence in two dead crows found in the Chicago area. The following year, the state's first human case and death from West Nile disease were recorded and all but two of the state's 102 counties eventually reported a positive human, bird, mosquito or horse. By the end of 2002, Illinois had counted more human cases (884) and deaths (64) than any other state in the United States.

Since then, Illinois and more specifically Chicago, has continued to suffer from multiple outbreaks of the WNV. From 2005 to 2016, a total of 1,371 human WNV cases were [reported](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0227160) within Illinois. Out of these total reported cases, 906 cases (66%) were from the Chicago region (Cook and DuPage Counties). 

## Problem Statement

Being hired by the division of Societal Cures In Epidemiology and New Creative Engineering (DATA-SCIENCE), with the above in mind, <b>our project aims at predicting outbreaks of the WNV</b>. This helps the City of Chicago and the Chicago Department of Public Health (CDPH) to allocate resources more efficiently and effectively toward preventing the transmission of this potentially deadly virus. Specifically, our model will use a combination of weather, time, and location feature to predict the presence of WNV within mosquito traps set up throughout Chicago. 

<b> Our project also aims to determine the best strategy for controlling the spread of the WNV</b>, as well as <b>discussing and justifying the various trade-offs that need to be made in implementing our model</b>.

## Objective
This is a supervised learning task, since the labels are provided (the expected output, i.e., binary representation of whether WNV was present within mosquito traps). 

We will be predicting two discrete class labels. More specifically, this is a binary classification problem with the ultimate goal -- to build a classifier to distinguish between just two classes, whether WNV was present in these mosquitos. 1 means WNV is present, and 0 means not present.

We will evaluate the performance of our model using AUC (Area Under Curve) score as the North Star metric. AUC score can be obtained by measuring the area under the receiver operating characteristic(ROC) curve. The ROC curve plots the true positive rate (TPR) against the false positive rate (FPR).  Our motivation is to increase the correct prediction (TP, TN) out of all the prediction, to detect correctly. The closer the AUC score to 1, the better the model is performing. 

## Datasets

The dataset, along with description, can be found here: [https://www.kaggle.com/c/predict-west-nile-virus/](https://www.kaggle.com/c/predict-west-nile-virus/).

- `train.csv` : the training set consists of data from 2007, 2009, 2011, and 2013.
- `test.csv`  : the test set is used to predict the test results for 2008, 2010, 2012, and 2014.
- `weather.csv`: weather data from 2007 to 2014.
- `spray.csv` : GIS data of spraying efforts in 2011 and 2013

Every year from late-May to early-October, public health workers in Chicago setup mosquito traps scattered across the city. Every week from Monday through Wednesday, these traps collect mosquitos, and the mosquitos are tested for the presence of West Nile virus before the end of the week. The test results include the number of mosquitos, the mosquitos species, and whether or not West Nile virus is present in the cohort.

<u>`train` & `test` dataset </u>

These test results are organized in such a way that when the number of mosquitos exceed 50, they are split into another record (another row in the dataset), such that the number of mosquitos are capped at 50. The location of the traps are described by the block number and street name. These attributes have been mapped into Longitude and Latitude in the dataset and are derived locations. For example, Block=79, and Street= "W FOSTER AVE" gives us an approximate address of "7900 W FOSTER AVE, Chicago, IL", which translates to (41.974089,-87.824812) on the map.

Some traps are "satellite traps". These are traps that are set up near (usually within 6 blocks) an established trap to enhance surveillance efforts. Satellite traps are postfixed with letters. For example, T220A is a satellite trap to T220. Not all the locations are tested at all times. Also, records exist only when a particular species of mosquitos is found at a certain trap at a certain time.

<u>`spray` dataset</u>

The City of Chicago also does spraying to kill mosquitos. Spraying can reduce the number of mosquitos in the area, and therefore might eliminate the appearance of West Nile virus.

<u>`weather` dataset</u>

It is believed that hot and dry conditions after wetter condition are more favorable for West Nile virus than cold and wet. The dataset is from National Oceanic and Atmospheric Administration (NOAA) of the weather conditions of 2007 to 2014, during the months of the tests.

Station 1: CHICAGO O'HARE INTERNATIONAL AIRPORT Lat: 41.995 Lon: -87.933 Elev: 662 ft. above sea level

Station 2: CHICAGO MIDWAY INTL ARPT Lat: 41.786 Lon: -87.752 Elev: 612 ft. above sea level

## Data Dictionary

| **Columns** | **Type** | **Dataset** | **Description** |
|---|---|---|---|
| **Id** | *integer* | test | The id of the record |
| **Date** | *datetime* | train/ test | Date that the WNV test is performed (YYYY-MM-DD) |
| **Address** | *object* | train/ test | Approximate address of the location of trap. This is used to send to the GeoCoder. | 
| **Species** | *object* | train/ test | The species of mosquitos |
| **Block** | *integer* | train/ test | Block number of address | 
| **Street** | *object* | train/ test | Street name |
| **Trap** | *object* | train/ test | Id of the trap |
| **AddressNumberAndStreet** | *object* | train/ test | Approximate address returned from GeoCoder |
| **Latitude, Longitude** | *float* | train/ test | Latitude and Longitude returned from GeoCoder |
| **AddressAccuracy** | *integer* | train/ test | Accuracy returned from GeoCoder |
| **NumMosquitos** | *integer* | train/ test | Number of mosquitoes caught in this trap |
| **WnvPresent** | *integer* | train/ test | Whether West Nile Virus was present in these mosquitos. (1 means WNV is present, while 0 means WNV is absent.) 
| **Date** | *datetime* | spray | The date of the spray (YYYY-MM-DD) |
| **Time** | *object* | spray | The time of the spray |
| **Latitude, Longitude** | *float* | spray | The Latitude and Longitude of the spray |
| **Station** | *integer* | weather | Weather station (1 or 2) |
| **Date** | *datetime* | weather | Date of measurement (YYYY-MM-DD)|
| **Tmax** | *integer* | weather | Maximum daily temperature (in Degrees Fahrenheit, F) |
| **Tmin** | *integer* | weather | Minimum daily temperature (in Degrees Fahrenheit, F) |
| **Tavg** | *object* | weather | Average daily temperature (in Degrees Fahrenheit, F) |
| **Depart** | *object* | weather | Departure from normal temperature (in Degrees Fahrenheit, F) |
| **DewPoint** | *integer* | weather | Average Dew Point temperature (in Degrees Fahrenheit, F) |
| **WetBulb** | *object* | weather | Average Wet Bulb temperature (in Degrees Fahrenheit, F) |
| **Heat** | *object* | weather | Heating Degree Days (season begins with July) |
| **Cool** | *object* | weather | Cooling Degree Days (season begins with January) |
| **Sunrise** | *object* | weather | Time of sunrise (calculated) |
| **Sunset** | *object* | weather | Time of sunset (calculated) |
| **CodeSum** | *object* | weather | Code of significant weather phenomena |
| **Depth** | *object* | weather | Snow/ice depth on the ground in inches, measured at 1200 UTC |
| **Water1** | *object* | weather | Water equivalent in inches, measured at 1800 UTC |
| **SnowFall** | *object* | weather | Total snowfall precipitation for the day (in inches and tenths) |
| **PrecipTotal** | *object* | weather | Total water equivalent precipitation for the day (in inches and tenths).  |
| **StnPressure** | *object* | weather | Average station pressure (in inches of hg) |
| **SeaLevel** | *object* | weather | Average sea level pressure (in inches of hg) |
| **ResultSpeed** | *float* | weather | Resultant wind speed (mph) |
| **ResultDir** | *integer* | weather | Resultant wind direction (degrees) |
| **AvgSpeed** | *object* | weather | Average wind speed (mph) |

More detailed descriptions for weather data can be found in [noaa_weather_qclcd_documentation]('./assets/noaa_weather_qclcd_documentation.pdf').

## Data Cleaning:
Getting sufficient information into the data is important to understand and approach this tack.

With the given 4 datasets, we have carried out initial analysis on each of them and produced the summary and list of cleaning works to be done.
These include 
- dropping duplicate rows, 
- imputing missing and zero values, 
- splitting strings and replace with correct format, 
- converting to right data types,
- dropping columns with high missing values
- creating more interpretable features

## EDA 

- Outbreak locations map
<img src="./assets/WNV Outbreak Locations.png" title="Wnv Outbreak Locations" width="800" height="300" />
From the map, we observed that WNV is more prevalent near bodies of water and O'Hare Airport. With this helpful visualization of WNV outbreak locations, CDPH and the City of Chicago can make more informed decisions on areas of hotspots and thus give a higher priority to spraying resources in these areas.

- Spray locations map
<img src="./assets/Spray Locations.png" title="Spray Locations" width="800" height="300" />
From the map, we have discovered that all the traps are spread out well geographically, however the spraying fails to fully overlap with the virus outbreak. This is a cause for concern and it can be because of multiple reasons e.g. - improper spraying practices, resources constraint. We will explore this further in the modelling and cost benefit analysis.

- Monthly NumMosquitos
From the month’s graph, we can see that the WNV cases started from July through and reduced in September, which is the summer season in Chicago.

When we look at the week’s graph, we can see that the WNV cases started from week 26 (between the end of June and early July) and decreased at week 40 (between the end of September and early October). Week 34 (around mid-August) has the highest cases every year.

Our first instinct from this information is that most of the WNV cases that happened around summertime must be related to the weather which encourages mosquito breeding. We will investigate the weather information later.

*The target is highly **imbalanced** with majority class (94.8%) to be without WNV.* This will be taken care of before modelling.

From the table and graph above, we can see that Culex Pipiens and Culex Restuans are the most captured mosquitos and the only mosquitos that will carry West Nile Virus, about 0.41 % of the total samples are carrying the virus. Thus, we will be focusing on both mosquitos' activities.

- Weather effect on Wnv
When the wet bulb temperature exceeds 59° Fahrenheit and dew point temperature exceeds 50° Fahrenheit, it will create a humid environment that is important to the mosquito's activities. Thus, we can see the more mosquitoes with WNV are detected.

When we look at the relationship between wet bulb temperature/ dew point temperature vs WNV present individually, we can see that there is a range showing that the WNV detected:
- wet bulb temperature - between 50° to 76° Fahrenheit
- dew point temperature - between 55° to 70° Fahrenheit

Based on the exploratory data analysis, we noted that not all mosquito species carry WNV. The two main vectors of WNV are CULEX PIPENS and CULEX RESTUANS, they contributed to 99.5% of the mosquitos captured in the traps. Summer is the season whereby WNV occurrences increase, starting from June, and peaks in August before declining slightly in September. While traps have been placed across Chicago, not all hotspots where WNV outbreaks occur have traps set up thoroughly. There is only spray information recorded in 2011 and 2013. The information shows that the spray area fails to fully overlap with the virus outbreak hotspots, indicating the cause for concern as a result of improper spray coverage.

## Feature Engineering
We've observed that our features generally have a pretty low correlation to `WnvPresent`, the strongest feature is `NumMosquitos` with a correlation of 0.2. [CDC](https://www.cdc.gov/westnile/resourcepages/mosqSurvSoft.html#:~:text=The%20simplest%20estimate%2C%20the%20minimum,goals%20of%20the%20surveillance%20program) defined the simplest and traditional estimate <b>Minimum Infection Rate (MIR)</b> which assumed that a positive pool contains only one infected mosquito (an assumption that may be invalid):

$$ \text{MIR} = 1000 * {\text{number of positive pools} \over \text{total number of mosquitos in pools tested}} $$

CDC has developed easy-to-use programs for calculating virus infection rate (IR) estimates from mosquito pool data using methods that do not require the assumption used in the MIR calculation.

$$ \text{IR} = {\text{number of infected mosquitos} \over 1000} $$

CDC encourages to incorporate virus infection rate (IR) into their mosquito-based evaluation of local virus activity patterns. At the county level or below, weekly tracking of mosquito IR can provide important predictive indicators of transmission activity levels associated with elevated human risk.

Unfortunately, our test data doesn't have the information we need to make this a usable feature. We discussed estimating the number of mosquitos based on total rows in the test set, but we ultimately decided that this was a slightly [<i>'hackish'</i> solution](https://www.kaggle.com/c/predict-west-nile-virus/discussion/14790). We'll drop NumMosquitos moving forward.

Our remaining features can be categorised as a mixture of 
1. time, 
2. weather (e.g. Temperature, Precipitation), 
3. location variables. 

Each of these variables has a low correlation of absolute 0.1 or less to our target. While we certainly could just go ahead with these features and jump straight into predictive modelling, a much better approach in the form of feature engineering is available. Without engineering, our models consistenly scored an AUC-ROC of approximately 0.5.

In this section, we'll look to <b>decompose and split our features</b>, as well as carry out <b>data enrichment</b> in the form of historical temperature records from the [National Weather Service](weather.gov). We'll also carry out a bit of polynomial feature engineering, to try and create features with a higher correlation to our target. 

Ultimately, we opted to drop these features. Dropping `year` has to no change in performance (we have `YearWeek` instead), and our other three polynomial features didn't give us significant increase in model AUC to justify the decreased interpretability of our model.

## Modelling 

In this section, we tested out a variety of predictive models including a Logistic Regression classifier and tree-based algorithms like AdaBoost. We carried out the following process:
- Train-test-split data
- Calculated baseline and benchmark models
- Fit model to training dataset
- Ran models on our data without using any over- or under-sampling techniques to benchmark performance
- Used the <b>Synthetic Minority Oversampling Technique (SMOTE)</b> to address the class imbalance within our target variable
- Carried out hyper-parameter tuning on our most promising models 
- Identified our top performing model based on ROC-AUC score

The class imbalance within our target variable highlights the issue of using accuracy or R<sup>2</sup> as a metric for our model. The training dataset is strongly biased towards samples where WNV is absent. This means that simply classifying every data point as absent of WNV would <b>net our model accuracy of almost 94.6%</b>. 

In this scenario, we need another metric to help us avoid overfitting to a single class. Using Area Under Curve (AUC) is a great alternative, as it focuses on the sensitivity (TPR) and specificity (TNR) of our model. To elaborate, AUC measures how true positive rate (recall) and false positive rate trade-off. This reveals how good a model is at distinguishing between a positive class and a negative class.

Using an AUC Reciever Operating Characteristic or AUC-ROC curve, <b>we can visually compare the true positive and false positive rates at a range of different classification thresholds to identify our best model</b>.

Our **Baseline** model's ROC AUC train or test score is 0.5.

Below are the classifiers that we used for comparing our models:
- LogisticRegression 
- RandomForestClassifier 
- GradientBoostingClassifier
- DecisionTreeClassifier
- ExtraTreesClassifier
- AdaBoostClassifier
- SupportVectorClassifier
          
Our initial analysis on **AUC-ROC** curve is obtained after comparing all of our models. Our non-boosting tree classifiers tend to have a sharp drop off in true positive versus false positive rate after a specific threshold. In comparison, our **Logistic Regression and Boosting models** seem to be performing better in terms of AUC.

Beyond focusing just on AUC which looks how good our modelling is at separating our positive and negative class, we also want to pay close attention to our model's ability to classify most or all of our minority class (which in this case is our positive class). Using a Precision-Recall AUC curve, we can look at the trade-off between precision (number out of true positives out of all predicted positives) and recall (number of true positives out of all predicted results).

### Results:
<img src="./assets/final_model" title="feature importantcy" width="800" height="300" />


