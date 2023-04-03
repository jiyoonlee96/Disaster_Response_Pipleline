#Import libraries
import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine
import nltk
import re
import pickle
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB

nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):

    """
    Loads dataset from SQLite database and define feature and target variables X and y
    
    Parameters :
    database_filepath : Filepath of the database  
    
    Returns :
    X : Features
    y : Target
    """

    #Load data from database
    engine = create_engine('sqlite:///InsertDatabaseName.db')
    df = pd.read_sql_table("disaster_messages", con=engine)

    #Define feature and target variables X and Y
    X = df['message']
    y = df.iloc[:,4:]
    
    return X, y


def tokenize(text):
    
    """
    Function to tokenize and lemmatize message data 

    Parameters : 
    text : a disaster message

    Returns :
    clean tokens : list of cleaned token in the message

    """

    #Get tokens from text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    #Clean tokens
    clean_tokens=[]
    
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
        
    return clean_tokens



def build_model():
   
    """
    Builds a machine pipeline that takes in the message colum as input and output classification results on the other 36 categories in the dataset
    (uses MultiOutputCalssifier to predict multiple target variables).

    Returns :
    cv : Classifier
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
            'tfidf__use_idf': (True, False),
            'clf__estimator__n_estimators': [50, 100],
            'clf__estimator__estimator__C':[1,2,5]
    }

    model = GridSearchCV(pipeline, param_grid=parameters)

    return model


def evaluate_model(model, X_test, y_test):
    
    """

    Parameters : 
    model : Classification model
    X_test : Test data features
    y_test : Test data target

    Returns :
    classficiation report for each column
    """
    y_pred = model.predict(X_test)
    
    for i, col in enumerate(y_test):
            print(col)
            print(classification_report(y_test[col], y_pred[:, i]))


def save_model(model, model_filepath):
    
    """
    Saves the model as a pickle
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()