# Titanic- Data Analysis, Visualization and Accuracy score calculation
![image](https://user-images.githubusercontent.com/126722476/224194133-9622c3aa-507b-4866-8884-d755e7fa4f98.png)

## Let's talk about the Titanic dataset

The Kaggle website for Titanic competetion provided 3 csv files which are train.csv, test.csv  and gender_submission.csv.

#### train.csv - 
This file contains the actual data of the passengers and their survival outcome.
I have trained the model with different classifiers like Linear SVM, Decisionn Tree, Logisitic Regression and Random Forest and selected the features using techniques like Chi-Square test, correlation parameters-heat map. 

#### test.csv-
This file contains the unseen data which doesn't give us the information about the survival outcome for the passengers. Using the model trained over the features I have predicted the survival outcome for the passengers. 

#### gender_submission.csv -
This file contains the data about the female passengers  and their survival outcome, as an example of what a submission file should look like.

#### Data Dictionary

**Variable** | **Definition** | **Key** 
-------------|----------------|--------
PassengerId  |Serial numbers which are uniques to each passenger|
Survived| Survival outcome| 0= didn't survive  1=survived|
Pclass|Ticket class| 1=Upper, 2=Middle, 3=lower|
Name|Name of passengers| |
Sex|Gender of passengers|Male,Female|
Age|Age of passengers in years||
SibSp|number of sibling/spouses on ship|0-None, 1 or more|
Parch|number of parents/children on ship|0-None, 1 or more|
Ticket|Ticket number||
Fare|Passenger fee||
Cabin|Cabin number||
Embarked|Embarkation port|C = Cherbourg, Q = Queenstown, S = Southampton|

