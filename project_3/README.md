# Project 3: Web APIs & NLP

## Executive Summary
Brand positioning and digital presence are two of the important success factors in brand battles nowadays. Companies are doing competitor analysis at various aspects in all manners feasible. There will be multiple subprojects running across multiple platforms from a nationwide community to complete the whole analysis. This project, as a subset of the whole project, focuses ONLY on the Reddit platform and aims to assist Dunkin' Brands Group Inc. - Dunkin' Donuts to know their company's online presence status and product reviews as compared to their largest competitor, Starbucks Corporation. 

The fact that a subreddit provides an unbiased platform for discussion on all things regarding a brand. Customers, Employees, Connoisseurs, and Executive Chefs are all welcome to join to celebrate, commiserate or inform. Each community supports general discussion of brands whatever their source. [(*source*)](https://www.reddit.com/r/DunkinDonuts/) 

In the first step of all, we utilize the latest Reddit posts from the respective brands and implement a classification model to allocate the posts to either 'Dunkin' Donuts' or 'Starbucks'. The classification model here is binary as the output variable is binary. After evaluating the various models, our top five models are **Logistic Regression**, **Extra Tree Classifier**, **Ridge Classifier**, **Gradient Boost Classifier**, and **Random Forest**. The **Logistic Regression Model** was chosen as the best model as it gave the best accuracy score.

The use cases of this classification model include that we can make use of unbiased posts to assess how the community perceives the Dunkin' Donuts brand as well as the competitor. Besides that, we also gain insights from the community to evaluate or retrospect our marketing strategies to determine the focus and pick up the latest trends. This subsequently helps in forecasting sales and demand, planning additional manpower during seasonal times, and even exploring more customization services.  

From the research, Starbucks seems to be more popular and has a more active community than Dunkin's. For both coffee chains, some trending topics are similar. Those topics are their services and new products. After performing sentiments and emotion analysis on the Reddit posts, 'iced coffee' and 'reward' seem to gather more positive sentiments. Thus, these two could be the focus of a marketing campaign. 

## Introduction & Problem Statement
Dunkin' has a digital presence across Twitter, Facebook, Instagram, and Pinterest and has launched many successful digital campaigns to attract new customers and increase sales. It has implemented a simple strategy to enhance its social media presence, namely, marketing a colorful and quirky personality online. [(*source*)](https://unmetric.com/brands/dunkin-donuts) 

At all times, Dunkin' is commited to leverage technology to provide consumer conveniences, such instance as the launch of integrated On-the-Go mobile ordering application with Google Assistant. [(*source*)](https://www.prnewswire.com/news-releases/dunkin-donuts-integrates-on-the-go-mobile-ordering-with-the-google-assistant-300613861.html) The popular Pumpkin Spice Signature Latte and fall range of beverages and snacks, which has released on August 17 and was just ended on September 13. [(*source*)](https://hypebae.com/2022/8/dunkin-donuts-fall-menu-pumpkin-spice-latte-coffee-cold-brew-release-info) Dunkin's group is very interested in knowing their brand online presence, their product reviews, and the effectiveness of marketing strategies as compared to their largest competitor, Starbucks. 

The first step would be to unveil the unbiased trending topics of these two big brands within Reddit. Subsequently, they are exploring ideas from the data across multiple platforms among the nationwide community, including product reviews, the share of voices, and mentioners. We are entrusted to develop a classification model to predict which class a post belongs to. At the same time, we can gain insights from the community to evaluate or retrospect our marketing strategies to determine the focus and to stay current with the latest trends by being consumer-centric and adapting to consumer insights. The customers get to enjoy a better and more pleasant experience with Dunkin'.

From the analysis, we identified the recent topics of interest related to the business and the community’s sentiments towards them. Following that, we provided recommendations for boosting their upcoming marketing campaign. That said, this would provide them with an indicative area of focus.

To approach this problem, this project aims to:
- Identify the trending topics from the subreddits of Starbucks and Dunkin Donuts
- What are the sentiments and emotions of the community in general and towards the topics/products
- Develop a Classification Model to distinguish Starbucks and Dunkin Donuts posts

### Key Questions
- What are the trending topics for each community?
- What is the best model to classify posts?
- Which products should we focus our marketing on?
- Regarding top topics, what are the community’s sentiments and emotions towards them?

### Data Science Process
- Data Collection
- Data Cleaning and EDA
- Pre-processing
- Modelling
- Model Evaluation
- Sentiments and Emotions Analysis

## Data Collection
We webscrapped data using Pushshift Reddit API. In this process, we intended to extract the last 2,500 posts from Dunkin Donuts and Starbucks subreddit respectively for analysis. The timeframe we set was before September 15, 2022 1137hr GMT+08:00.

2,498 out of 2,500 Dunkin Donuts posts were extracted. Most posts were extracted dated from 3 February 2022 to 15 September 2022.

2,499 out of 2,500 Starbucks posts were extracted. Most posts were extracted dated from 23 August 2022 to 15 September 2022. 

## Data Cleaning and EDA

### Data Cleaning
We used the title, title_selftext, subreddit, and created utc for our analysis. During cleaning of the title and title_selftext columns, we removed noises in our data such as html links, symbols ,and any markdown language like '/n'. We converted Emojis to text. 

Given the fact that we do not always remove the stop words. The removal of stop words is highly dependent on the task we are performing and the goal we want to achieve. For example, if we are training a model that can perform the sentiment analysis task, we might not remove the stop words. [(*source*)](https://towardsdatascience.com/text-pre-processing-stop-words-removal-using-different-libraries-f20bac19929a#:~:text=We%20do%20not%20always%20remove,was%20not%20good%20at%20all.%E2%80%9D) For modelling, we remove stop words from the corpus, the stop words list is customized to include words such as DunkinDonuts, Starbucks or any similar words to those. Most documents have words less than 100 words. Small groups of documents have more than 200 words. We only kept documents with at least 2 words as single-word documents do not seem to be useful for modelling. We assessed both Stemming and Lemmatizing. Stemming was used as it reduced the number of features by a greater magnitude. There are 7381 words remaining after stemming. Most of the top words remained after cleaning - except the DunkinDonut(s), starbucks or similar words.

### EDA
2773 unique words in Dunkin Donuts corpus. 1777 unique words in Starbucks corpus.These words appeared only once in corpus. Total 0 unique words remain in the combined corpus. The top occurring unigrams common for both datasets are 'like', 'order', 'drink' ,and 'coffee'. Their respective specific words are 'dunkin', 'donuts', 'starbucks'. The top bigram words for DunkinDonuts are 'cold brew', 'iced coffee', 'iced latte', 'frozen coffee', 'dunkin donuts'. The top bigram words for Starbucks are 'dress code', 'cold brew', 'iced coffee', 'pumpkin spice', 'brown sugar'. 

Both CountVectorizer (most frequent) and TDIDF (most important) return mostly the same words. Note that TF-IDF is better than Count Vectorizers because it not only focuses on the frequency of words present in the corpus but also provides the importance of the words. We can then remove the words that are less important for analysis, hence making the model building less complex by reducing the input dimensions. [(*source*)](https://www.linkedin.com/pulse/count-vectorizers-vs-tfidf-natural-language-processing-sheel-saket#:~:text=TF%2DIDF%20is%20better%20than,by%20reducing%20the%20input%20dimensions.) Judging by the count of posts and period of the 2,500 post, for each subreddit, Starbucks looks like it is a more popular coffee chain. During the last few months, Starbucks had more posts on average.

![wordcloud]("./assets/wordcloud.png")

## Preprocessing and Modelling
We imported clean data, convert our text column into numerical values using either CountVectorizer or TF-IDF Vectorizer.

We held 30% of whole data for testing, which is not exposed to any fitting from the modelling. We generate predictions against this dataset. Target Class Labels: 
- Dunkin Donuts : 1
- Starbucks : 0

Our target variable is in binary value, this is a binary classification problem. We will evaluate the classification models to get the best model. The baseline model is simply the normalize value of majority class. we have defined our null/ baseline model to be the value of majority class. As the first model, we will go with Naive Bayes model after performing CountVectorizer and compare other models against the baseline.

Subsequently, we used a TF-IDF vectorizer for our model factors in the importance of the word relative to other documents in the corpus. This reduces the effects of words that are frequent in both subreddits but has no predictive power. Also, I will proceed with a unigram and bigram bag of words. As seen from our EDA section, many of the top games and features of each console appear in our top bigram.

## Evaluation and Conceptual Understanding
We evaluated 4 models, namely **Logistic Regression(lr)**, **Random Forest Classifier(rf)**, **Support Vector Machine(svm)** & **Gradient Boosting Classifier**. We tuned the hyperparameter to get the best results. 
    
Score Summary 
|Model|Logistic Regression|Random Forest Classifier|Support Vector Machine|Gradient Boosting Classifier|
|---|---|---|---|---|
| CV score | 0.7722 | 0.7447 | 0.7378 | 0.7580 |
| Train score | 0.7763 | 0.7607 | 0.7529 | 0.7510 | 
| Test score | 0.7808 | 0.7726 | 0.7502 | 0.7611 |

From our results, Logistic Regression gives the best accurary of 0.78. It calculates the probability of the target outcome for each document using the log-odds. All models does not show any sign of overfitting, as there is not a significant difference. The result is much better than the accuracy of baseline model (0.50). Generally, metrics alone are not the only criteria we should consider when finalizing the best model for production. Other factors to consider include training time, standard deviation of kfolds etc. A Logistic Regression is computationally inexpensive compared to Random Forest, Support Vector Machine and Gradient Boosting. Besides, Logistic Regression is easy to interpret. 

In summary, **Logistic Regression** is the best model.    
    
## Sentiment and emotion Analysis
Model used for sentiment analysis: cardiffnlp/twitter-roberta-base-sentiment [(*source*)](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
Model used for emotion analysis: Bert-base-uncased-emotion. [(*source*)](https://huggingface.co/bhadresh-savani/bert-base-uncased-emotion)

In Summary, the results from our sentiment and emotion analysis could be useful to answer the problem statement. General speaking, the results from sentiments analysis seem to be a better representative than the emotions analysis due to the neutral nature of most posts. Reward is the topic that generate more positive sentiments for both coffee chains. However, service seems to be a less well-received topic, possibly due to complaints or issues. This might also true for topic local Dunkin. Starbucks fall launch seems to be well received and it gains favor attention within community. These are some insights that could be useful for our client's marketing campaign.
    
## Conclusion    
Based on our research, we can conclude that Starbucks is the more popular among the two. Nevertheless, Dunkin' has quite huge and stable community and is able to further expand to more potential markets. The topics of interest for Dunkin seems to be reward system and some of their products. Iced coffee and butter pecan have been given more positive sentiments. Starbucks upcoming fall launch also seems to be well-received. 

Our sentiments and emotions analysis also indicate that most community posts are neutral in nature. In order to create a solution for the the website moderator, we will use a Logistic Regression Model that is trained on variety relevant sorting topics reddit posts to distinguish the topics.   

## Limitations
Both coffee chains tend to have very similar topics as well as same words that appear prominently in many of their posts as they are both very similar in nature. These words that appear commonly in both topics - such as coffee, beverages and reward? could be factors that limit our predictions.

Emotions analysis seem to be ineffective in giving accurate outcomes due to the neutral nature of most post on reddit and might not be effective in predicting titles that relate to questions, news and facts.
    
## Recommendation
Marketing Campaign:
- Allocate more resources to iced coffee and butter pecan, these are generally well-received and could be main focus for marketing
- Try partner with influencer to promote new launch for upcoming fall
- Prioritize attractive Reward 

Social Media categorisation:
- A Logistic Regression Model can be implemented to help with the classifying of documents

Future Work:
- Retrieve more data from Reddit such as Comments, Upvote and Hot
- Retrieve data from other sources such as other Tiktok, Twitter and Facebook
- Utilising GPU/cloud service to process a bigger data volume