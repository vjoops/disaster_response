# Disaster Response Pipeline Project
### Project Description:
This project implement an ETL (Extract, Transform, Load) for data coming from Figure 8.  
The data  consist of two files:
<li>disaster_messages.csv: this contains all messages that are received by the system.
<li>disaster_categories.csv: this contains all the category that each message belongs to.

At the end of ETL, the application loads data into Messages table in the sqllite database.  

After the data is loaded into database, a machine learning training, fit, and evaluate algorithm 
is run to create a machine learning model to automatically classify the message.

Application shows some visualization of how data so far looks like and give user a chance to enter 
a new message and the application shows what that message classify to.
 
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
