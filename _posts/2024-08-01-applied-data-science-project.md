---
## Background and Business Goal

Since its inception in 1969 in Paris, Sephora has expanded globally and revolutionized beauty retailing with a focus on luxury and expertise. It offers a wide selection of skincare products and solutions, ranging from makeup, skincare, hair care and beauty tools.  

Its dedication to empowering customers through personalized service and expert advice is central to Sephora's success. Ongoing competition and market saturation challenges mean Sephora must constantly distinguish itself from other competitors through its unique brand identity and offering. Through innovations with digital tools and enhanced personalized recommendations based on customer preferences, it remains dedicated to its mission to elevate the beauty retail experience and lead in the industry. 
This project aims to gather valuable insights into customer satisfaction and sentiment by analysing helpful reviews and features. These include identifying best-selling items, product effectiveness through reviews and ratings, and a recommendation system for cross-selling to increase sales and customer satisfaction.   

## Project Background

The primary business objectives are to determine whether customer feedback is positive or negative, and to use these insights to enhance decision-making processes. This involves identifying common issues mentioned in reviews, which can inform potential areas for product improvement. Additionally, the project aims to monitor changes in sentiment over time and correlate them with product developments or changes in marketing strategies. The data mining goal is to classify customer feedback into sentiment categories, which will provide actionable insights for improving customer experience and satisfaction.

## Work Accomplished

### Text Preprocessing

#### Data Collection and Preparation
The project involves analyzing a dataset from Kaggle that includes Sephora product and skincare reviews, comprising one product info file and five review files. These five review files are concatenated into a single DataFrame named "combined_df_reviews" for further analysis.

![image](https://github.com/user-attachments/assets/f5b73228-3788-4444-9c39-82a2aa092c4c)

The key focus during data preparation is on the 'review_text' and 'review_title' columns, which are combined into a new column called 'combined_review' to facilitate sentiment analysis. Other columns not relevant to the analysis, such as 'author_id' and 'submission_time,' are removed to streamline the dataset, focusing only on the variables necessary for extracting insights from customer reviews.

![image](https://github.com/user-attachments/assets/b31b5ea1-2992-45e0-b782-4098f5cdca96)

#### Tokenisation
In tokenisation, the text is divided into individual words or tokens, which is a crucial step in sentiment analysis as it helps in breaking down customer reviews into manageable units for analysis. This process supports the business goal of understanding customer sentiment by allowing the model to analyze each word's contribution to the overall sentiment of the text. In this project, the nltk.tokenize module from the Natural Language Toolkit (NLTK) library is imported, and the sent_tokenize function is applied to the review column. This function splits the review text into a list of sentences, making it easier to process and analyze each sentence's sentiment individually, thereby contributing to a more accurate understanding of customer feedback.

![image](https://github.com/user-attachments/assets/d8f5ba22-3763-4805-b83d-81acf27afd6a)
 
#### Normalisation
The text is converted to lowercase to ensure uniformity and reduce complexity of the model. Each word token is converted to lowercase using the .lower() method.

#### Stop Words Removal
The words that do not carry significant meaning are removed. The ‘stopwords’ module is being imported from the nltk.corpus package. The module provides predefined lists of stop words. 

![image](https://github.com/user-attachments/assets/565eae60-7121-4751-9236-b41bdb74b398)

#### Punctuation Removal
Punctuation marks do not typically carry meaningful information regarding the sentiment or content of the text. Punctuations are removed using ‘string.punctuation’ module.

#### Stemming
Stemming is a text normalisation technique used in NLP to reduce words to their root form, to treat related words with the same base form as the same word to reduce complexity of the model. PorterStemmer class from the gensim library is imported to apply stemming algorithm to each word.

#### Lemmatisation
Lemmatisation is a technique that reduces words to their base or dictionary form, taking into account the word's context and part of speech. It improves the accuracy of text processing in sentiment analysis by ensuring that different forms of a word are standardised. 

The WordNetLemmatizer class from the nltk,stem.wordnet module is imported. The WordNetLemmatizer uses the WordNet lexical database to perform lemmatisation.
There will be two arguments being passed to the Wordnet Lemmatizer:
- The word to be lemmatised (word).
- The part of speech (POS) such as verbs, nouns and adjectives.

#### Final Word Cloud after Text Preprocessing
![image](https://github.com/user-attachments/assets/d256632a-0193-468c-b1a8-7b2c0241cc3c)

##### Key Themes
The largest words like "skin", "feel", "love", "moistur", "great", "good", "work", "hydr", and "face“ suggest that these are the most frequently discussed topics in customer reviews. 
>> Recommendation: Sephora can focus on these themes to understand what customers are specifically saying about them. For example, the word moisture seems to be drawing a lot of attention from the customers.

##### Customer Experiences
Words like "love", "great", "good", "amazing", "perfect", "wonderful", "best", "smooth", "soft", "nice", "glow", and "recommend" suggest positive experiences.
>> Recommendation: Sephora can consider analysing these positive reviews to identify products that customers love the most and highlight these in marketing campaigns to attract more customers. 
For example, focus more on glowing products.

##### Areas for Improvement
Words like "dry", "oily", "sensitive", and "breakout" might indicate issues customers face with the products.
>> Recommendation: Investigate these negative mentions to identify common problems. If "dry" is frequently mentioned, customers might be experiencing dryness from certain products. Consider formulating more how to mitigate these issues such as providing hydrating versions.

##### User Experience
Words related to application and use, such as "apply", "use", "smooth", and "easy", can provide insights into the user experience.
>> Recommendation: Improve product formulations based on feedback related to application and user experience. 
For example, if "apply" is often mentioned with ease or difficulty, it can indicate how user-friendly the product is.

## Word Representation

Word Representation covers essential techniques for converting text data into numerical vectors for sentiment analysis. It highlights four main methods under Vector Generation, Word Embedding and Feature Extraction. 

### Vector Generation - Bag-of-Words
The Bag-of-Words (BoW) method utilizes CountVectorizer to convert text into a matrix of token counts.

### Vector Generation - TF-IDF
TF-IDF uses TfidfVectorizer to transform text into weighted features based on term frequency-inverse document frequency. 

### Word Embedding (Word2Vec) and Feature Extraction (Parts-of-Speech Tagging)
Word Embedding involves training a Word2Vec model on tokenized documents to create dense vector representations of words. Lastly, Parts-of-Speech Tagging uses pos_tagging_filter to tag and filter words by their part of speech, focusing on nouns, verbs, adjectives, and adverbs. 

### The final output maps each unique word to a unique integer index, creating numerical feature vectors that are crucial for machine learning models in sentiment analysis.
![image](https://github.com/user-attachments/assets/df8554e7-8cb8-4b28-8100-804429f6b649)

## Modelling
In the modeling selection process for sentiment analysis, three different models were evaluated: Decision Tree, Support Vector Machine (SVM), and Naive Bayes. 

### Decision Tree
![image](https://github.com/user-attachments/assets/231de95f-693e-4404-b010-600cdde2a4d4)
#### Accuracy: 0.94275 
#### Weighted Avg F1-Score: 0.94
Lowest accuracy compared to Support Vector Machine and Naives Bayes. Better at capturing ‘negative’ class compared to Naives Bayes.

### Support Vector Machine
![image](https://github.com/user-attachments/assets/b0da1717-9af6-4d97-ad7a-bcf03a6807c0)
#### Accuracy: 0.95885
#### Weighted Avg F1-Score: 0.95 
Highest accuracy compared to Decision Tree and Support Vector Machine, strong performance on ‘neutral’ and ‘positive’ classes. 
Higher accuracy compared to Decision Tree. However, it has weak performance on ‘negative’ class. 

### Naives Bayes
![image](https://github.com/user-attachments/assets/aa00fa13-69c0-4db4-ad2f-fc6eaffcf809)
#### Accuracy: 0.95545 
#### Weighted Avg F1-Score: 0.94

Chosen Model: The Support Vector Machine (SVM) was selected due to its highest achieved accuracy of 0.95885, demonstrating its superior overall performance, particularly on the majority classes. This high accuracy suggests that the SVM model is well-suited for the task, offering strong predictive capabilities across the dataset.

### Hyperparameter Tuning

Hyperparameter tuning for the SVM model was conducted using GridSearchCV from scikit-learn, with the goal of optimizing the model's performance. The best parameters identified were {'kernel': 'linear'}, resulting in an improved accuracy of 0.9678. The classification report for the optimized SVM model shows strong performance across different classes, with a weighted average precision, recall, and F1-score all at 0.97. Notably, the model achieved a perfect precision and near-perfect recall for the neutral class, indicating highly reliable performance in that category.

![image](https://github.com/user-attachments/assets/7aa26870-c846-48c6-8eb1-576545b9b95a)

### Modelling on Unseen Data

![image](https://github.com/user-attachments/assets/70acf466-d340-404e-b172-2e5965159838)
>> Predicted Sentiment: Positive

![image](https://github.com/user-attachments/assets/eaf92cbd-5bc7-4065-badd-5529d56e53ab)
>> Predicted Sentiment: Negative

![image](https://github.com/user-attachments/assets/7fc8c6d0-01ad-4272-a061-2c370bc29390)
>> Predicted Sentiment: Negative

The model is applied to unseen data to effectively evaluate its performance. It demonstrates strong predictive capabilities, accurately identifying the sentiment of reviews. For instance, a review mentioning a 'horrible smell,' which clearly indicates a negative sentiment, is correctly classified as negative by the model.

## Evaluation

### Negative Sentiment
Terms like "smell," "dry," and "expensive" suggest common concerns or complaints related to the use of skincare and beauty products.

![image](https://github.com/user-attachments/assets/5baa0dc8-7652-404e-a364-84016e7db7f5)

### Positive Sentiment
Words like "great," "good," "nice," and "gentle" suggest satisfaction with product performance, quality, and overall experience. 

![image](https://github.com/user-attachments/assets/d7141cb0-ac96-4b3c-acb3-c301359330e4)

## Recommendation and Analysis
Explain the analysis and recommendations

### Negative Sentiment
#### Customer Experiences
Negative reviews mention dissatisfaction with products related to words like "smell,“ and "dry”.
Recommendations: Sephora can consider to conduct a review of these product lines focusing on concerns like dryness or irritation. 
#### Quality Control
Words like "bad," "expensive," and "disappointed" suggest that customers might feel some products are overpriced relative to their performance. 
Recommendation: Enhancing quality control and ensuring that all products meet high standards could help mitigate this sentiment.

### Positive Sentiment
#### Promote Top Products
Products that are frequently associated with positive words like "love," "great," and "amazing“
Recommendation: Sephora can consider emphasize in marketing campaigns as they resonate well with customers.
#### Recommendations
Frequent use of "recommend”. 
Recommendations: This suggests that satisfied customers are sharing their experiences with others, contributing to a positive brand reputation. Sephora can rewards points to customers that recommends products to their friends.

## AI Ethics

### Privacy
The project involves analysing user reviews of Sephora products which may contain personal information, therefore data used needs to be anonymised and users are consented before data is being used for analysis. Implementing encryption and access control measures to protect data.
### Fairness
Sentiment analysis models can show bias in data if training data is skewed towards certain demographics, such as the race of majority people and which can lead to unfair outcomes.
### Accuracy
Model need to be accurate and reliable to prevent misclassification of sentiments, leading to incorrect business decisions that may affect Sephora's revenue. Therefore, rigorous testing is required to validate the model's performance.
### Accountability
If Sephora is adopting ML for business decisions, the company needs to be accountable of the consequences and put a process in place.
### Transparency
Sentiment analysis models need to be transparent to allow stakeholders understand how decisions are made. Sephora needs to let stakeholders understand the objectives behind the analysis and how it will inform business or customer decisions.

## Source Codes and Datasets
[https://github.com/nicolewps/ITD214_Applied-Data-Science-Project]
### Dataset: 
[https://drive.google.com/drive/folders/18rFmz3ILGTLsMftKmjpSaBToO4Txn98r?usp=sharing]
