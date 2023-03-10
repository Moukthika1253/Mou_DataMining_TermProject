# Titanic- Data Analysis, Visualization and Accuracy score calculation
![image](https://user-images.githubusercontent.com/126722476/224194133-9622c3aa-507b-4866-8884-d755e7fa4f98.png)

## Let's talk about the Titanic dataset

The Kaggle website for Titanic competetion provided 3 csv files which are train.csv, test.csv  and gender_submission.csv.

#### train.csv 
This file contains the actual data of the passengers and their survival outcome.
I have trained the model with different classifiers like Linear SVM, Decisionn Tree, Logisitic Regression and Random Forest and selected the features using techniques like Chi-Square test, correlation parameters-heat map. 

#### test.csv
This file contains the unseen data which doesn't give us the information about the survival outcome for the passengers. Using the model trained over the features I have predicted the survival outcome for the passengers. 

#### gender_submission.csv 
This file contains the data about the female passengers  and their survival outcome, as an example of what a submission file should look like.

## Data Dictionary

**Variable** | **Definition** | **Key** 
-------------|----------------|--------
PassengerId  |Serial numbers which are uniques to each passenger|
Survived| Survival outcome| 0= didn't survive  1=survived|
Pclass|Ticket class| 1=Upper, 2=Middle, 3=lower|
Name|Name of passengers| |
Sex|Gender of passengers|Male,Female|
Age|Age of passengers in years.Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5||
SibSp|number of sibling/spouses on ship.Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife|0-no spouse|
Parch|number of parents/children on ship.Parent = mother, father
Child = daughter, son, stepdaughter, stepson|0-have only Nanny|
Ticket|Ticket number||
Fare|Passenger fee||
Cabin|Cabin number||
Embarked|Embarkation port|C = Cherbourg, Q = Queenstown, S = Southampton|

I have imported necessary libraries from the data from train.csv and test.csv into dataframes training_data and testing_data respectively through code 

### Sample data

Using the code from [kaggle] I have displayed the top 5 rows from train and test data.

|**Train data**|**Test data**|
|---|---|
|![image](https://user-images.githubusercontent.com/126722476/224204303-98e381a3-c718-4f15-b240-9a9cbd848d68.png)|![image](https://user-images.githubusercontent.com/126722476/224204389-72dc69a9-a315-485b-9e0f-6a64fd4a5625.png)|

## Data Pre-processing

The give Titanic data has imbalanced data and if we train the model without cleaning the data, the predictions wouldn't be that accurate. 
So in order to increase the performance of the model, consistency of data, making it balanced and to improve the accuracy of the predictions I have performed following data pre-processing techniques.

### Filling out the missing values in train, test data with their Mean and Mode

I found the missing values from both train and test data set and summed them up referring the code from [https://practicaldatascience.co.uk/data-science/how-to-use-isna-to-check-for-missing-values-in-pandas-dataframes] 

### Before fixing missing values

|Missing values in training_data|missing values in testing_data|
|---|---|
|![image](https://user-images.githubusercontent.com/126722476/224204551-8bb670f6-b722-43da-9112-8d9f5034d671.png)|![image](https://user-images.githubusercontent.com/126722476/224204593-cdfb0a69-2013-4438-8c95-dd5cac6c5ec5.png)|

From the above table we can tell that in training data columns Age, Cabin and Embarked have missing values. In test data, Age, Fare and Cabin have missing values.

### After fixing missing values
One of the better ways to deal with missing data is to fill them with their mean/median if the data is numerical and mode if the data is categorical. Since we have missing values in both categorical and numerical data I have filled them with the Mode(most repeating value) in Cabin and Embarked columns, with Mean(average) in Age, Fare columns. .

'''
training_data['Age']=training_data['Age'].fillna(training_data['Age'].mean())
training_data['Fare']=training_data['Fare'].fillna(training_data['Fare'].mean())
training_data['Cabin']=training_data['Cabin'].fillna(training_data['Cabin'].mode()[0])
training_data['Embarked']=training_data['Embarked'].fillna(training_data['Embarked'].mode()[0])
testing_data['Age']=testing_data['Age'].fillna(testing_data['Age'].mean())
testing_data['Fare']=testing_data['Fare'].fillna(testing_data['Fare'].mean())
testing_data['Cabin']=training_data['Cabin'].fillna(testing_data['Cabin'].mode()[0])
testing_data['Embarked']=testing_data['Embarked'].fillna(testing_data['Embarked'].mode()[0])
'''

Referred above code from [https://vitalflux.com/pandas-impute-missing-values-mean-median-mode/]



|training_data|testing_data|
|---|---|
|![image](https://user-images.githubusercontent.com/126722476/224204652-ab8a8fd9-0578-4a3b-8c1a-dd11a73235eb.png)|![image](https://user-images.githubusercontent.com/126722476/224204684-7937212b-c57a-46cf-8335-3f57adf216af.png)|




