import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Downloading NLTK stopwords for English
nltk.download('stopwords')

# Printing the stopwords in English
print(stopwords.words('english'))

# Loading the news dataset into a Pandas DataFrame
df_news = pd.read_csv('data.csv')

# Displaying the shape and first 5 rows of the dataset
print(df_news.shape)
print(df_news.head())

# Checking for missing values in the dataset
print(df_news.isnull().sum())

# Replacing null values with empty strings
df_news = df_news.fillna('')

# Merging the author name and news title into a new column 'content'
df_news['content'] = df_news['author'] + ' ' + df_news['title']
print(df_news['content'])

# Performing text preprocessing: stemming and removing stopwords
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

df_news['content'] = df_news['content'].apply(stemming)
print(df_news['content'])

# Separating data and labels
X = df_news['content'].values
Y = df_news['label'].values
print(X)
print(Y)
print(Y.shape)

# Converting textual data to numerical data using TF-IDF Vectorization
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

# Splitting data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=20)

# Training the model: Logistic Regression
model = LogisticRegression()
model.fit(X_train, Y_train)

# Accuracy on training data
Y_train_pred = model.predict(X_train)
training_accuracy = accuracy_score(Y_train, Y_train_pred)
print('Accuracy on training data: ', training_accuracy)

# Accuracy on test data
Y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, Y_test_pred)
print('Accuracy on test data: ', test_accuracy)

# Making a predictive system
X_new = X_test[3]  # Example data for prediction

prediction = model.predict(X_new)
print(prediction)

if prediction[0] == 0:
    print('The news is Real')
else:
    print('The news is Fake')
