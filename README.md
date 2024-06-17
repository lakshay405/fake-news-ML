# fake-news-ML
Fake News Detection using Logistic Regression and Text Processing
This project focuses on detecting fake news articles using machine learning, specifically Logistic Regression, after preprocessing the text data using Natural Language Processing (NLP) techniques.

Dataset
The dataset (data.csv) consists of news articles with labels indicating whether each article is real (0) or fake (1).

Workflow
Text Preprocessing:

Download NLTK stopwords for English and perform basic text cleaning operations:
Convert text to lowercase.
Remove non-alphabetic characters.
Tokenize the text and apply stemming using Porter Stemmer.
Remove stopwords from the text.
Data Loading and Preprocessing:

Load the news dataset into a Pandas DataFrame (df_news).
Display dataset shape, first few rows, and check for missing values.
Merge author names and news titles into a new column (content) for analysis.
Feature Extraction:

Convert textual data (content) into numerical features using TF-IDF Vectorization, which assigns weights to words based on their frequency and inverse document frequency.
Model Training and Evaluation:

Separate features (X) and target (Y) from the preprocessed dataset.
Split the data into training and testing sets using train_test_split.
Initialize and train a Logistic Regression model to classify news articles as real or fake.
Evaluate the model's performance using accuracy scores on both training and testing sets.
Prediction:

Demonstrate the model's prediction capability on example data from the test set.
Output the prediction result based on the predicted class.
Libraries Used
numpy and pandas for data manipulation and analysis.
nltk for text preprocessing (stopwords, PorterStemmer).
sklearn for model selection (LogisticRegression), evaluation (train_test_split, accuracy_score), and feature extraction (TfidfVectorizer).
Conclusion
This project demonstrates the application of Logistic Regression for detecting fake news based on textual content. By leveraging NLP techniques and TF-IDF Vectorization, the model can effectively distinguish between real and fake news articles, providing a valuable tool for media and journalism to combat misinformation.
