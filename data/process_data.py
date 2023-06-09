#Import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
   
    """
    Load messages dataframe and categories messages and merge them into one dataframe

    Parameters:
    messages_filepath: filepath of messages.csv
    categories_filepath: filepath of categories.csv

    Returns:
    df : messages dataframe and categories dataframe merged into one dataframe 
    """
    
    #load messages dataset
    messages = pd.read_csv('messages.csv') 
    
    # load categories dataset
    categories = pd.read_csv("categories.csv")
    
    # merge datasets
    df = messages.merge(categories, how='left', on=['id'])
    
    return df



def clean_data(df):

    """
    Cleans the dataframe by splitting columns, coverting category values to numeric,, and removing duplicates.  

    Parameters: merged dataframe of messages and categories

    Returns: cleaned version of merged dataframe
    """
    
    #Split categories into separate category columns
    #Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    #Select the first row of the categories dataframe
    row = categories.head(1)
 
    #Extract a list of new column names for categories.
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0,:]
 
    #Rename the columns of `categories`
    categories.columns = category_colnames  
    
    #Convert category values to just numbers 0 or 1
    for column in categories:
        #Set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
        #Convert column from string to numeric
        categories[column] = categories[column].astype(int)
    #Convert all categories to binary
    categories['related'] = categories['related'].astype('str').str.replace('2', '1')
    categories['related'] = categories['related'].astype('int')
   
    #Replace categories column in df with new category columns.
    #Drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    #Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    #Remove Duplicates
    #Drop duplicates
    df.drop_duplicates(inplace=True)

    return df



def save_data(df, database_filename):
   
    """
    Save the clean dataset into an sqlite database

    Parameters:
    df : Pandas dataframe object containing The cleaned data
    database_filename : filename of database.
    """
    
    #Save the clean dataset into an sqlite database
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace')



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()