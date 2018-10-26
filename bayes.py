# Importing required libraries and tools. Note: Tfidf vectorizer is used because it can remove stop words which will greatly reduce the number of features.
import numpy as np
import pandas as pd 
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Creating a dataframe; Assigning the column separator to be a tab indent; Labelling the columns.
df = pd.read_csv('spam.txt', sep = '\t', names = ['Classification', 'Message'])

# Replacing 'spam' with a 0 & 'ham' with a 1. This saves memory and makes computations easier down the line.
df.loc[df['Classification']=='spam','Classification']=0
df.loc[df['Classification']=='ham','Classification']=1

# Split the data into its independent and dependent components.
X = df['Message']
Y = df['Classification']

# Setup the tfidf vectorizer and the split the data into training and test sets. 
# min_df disregards anything below the specified word frequency. stop_words disregards common english words like 'the' 'are'...etc.
vec = TfidfVectorizer(min_df=1, stop_words='english')
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size=0.25)

# Create a tfdif vectorizer array with the training data.
X_train_vec = vec.fit_transform(X_train)
# To see the values assigned:
# X_vec_array = X_train_vec.toarray()

# Set up the classifier and the vectors to be used in classification.
mnb = MultinomialNB()
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')
X_test_vec = vec.transform(X_test) # not fit_transform here because we are matching the fit made by the training data.
mnb.fit(X_train_vec, Y_train)

# Predictions.
pred = mnb.predict(X_test_vec) # Creates an array with the predicted values: 1 for ham and 0 for spam.
result = mnb.score(X_test_vec, Y_test)
print(result) # Result = 97%.
