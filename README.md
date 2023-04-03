# Disaster_Response_Pipleline

## Summary
In this project, I applied my data enginnering skills to analyze disaster data from Appen (formally Figure 8) to build a model for an API that classifies disaster messages. The dataset used contains real messages that were sent during disaster events. I created a machine learning pipeline to categorize these events so that one can send the messages to an appropriate disaster relief agency.


## Project Components
There are three components you'll need to complete for this project.

### 1. ETL Pipeline
In a Python script, process_data.py, is a data cleaning pipeline that:

- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

### 2. ML Pipeline
In a Python script, train_classifier.py, is a machine learning pipeline that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

### 3. Flask Web App
Web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data. 


## File Structure
- app\
| - template\
| |- master.html  # main page of web app\
| |- go.html  # classification result page of web app\
|- run.py  # Flask file that runs app

- data\
|- disaster_categories.csv  # data to process\
|- disaster_messages.csv  # data to process\
|- process_data.py\
|- InsertDatabaseName.db   # database to save clean data to

- models\
|- train_classifier.py\
|- classifier.pkl  # saved model

- README.md


## Instructions
To run the app : 
1. Run the following commands in the project's root directory
  - to run the ETL pipeline `python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`\
  - to run the ML pipeline `python train_classifier.py ../data/DisasterResponse.db classifier.pkl`\
2. Run the following command in the app's directory
  `python run.py`
4. Go to http://10.0.0.173:3000/
