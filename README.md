layout: page
title: "Titanic- Data Analysis, Visualization and Accuracy score calculation"
permalink: /titanic dataset

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

![image](https://user-images.githubusercontent.com/126722476/224233893-9f6c0088-d485-4f96-ad77-a33b6fa315c7.png)


Referred above code from [https://vitalflux.com/pandas-impute-missing-values-mean-median-mode/]



|training_data|testing_data|
|---|---|
|![image](https://user-images.githubusercontent.com/126722476/224204652-ab8a8fd9-0578-4a3b-8c1a-dd11a73235eb.png)|![image](https://user-images.githubusercontent.com/126722476/224204684-7937212b-c57a-46cf-8335-3f57adf216af.png)|


## Data Encoding - Binary Encoder

Data Encoding is one of the pre-processing techniques. The encoding process involves converting the categorical data into numerical data. This is essential since majority of the algorithms need the data to be numerical and it also helps in improving the performance of the learning model as it can interpret the relationship between features and target variable in a better way. Therefore, I converted the categorical data (Name, Sex, Cabin, Ticket, Embarked) to numerical data in both training and test datasets using category_encoders library by referring code from https://pbpython.com/categorical-encoding.html

![image](https://user-images.githubusercontent.com/126722476/224235799-285c35f9-8a3a-4a5c-86f4-582102017b6f.png)

Binary encoder is a combination of OneHot Encoder and Hash Encoder. In OneHot Encoder the categorical data in nominal form is converted to binary values by creating new dummy variables. The Hash Encoder does the same but encodes them using hashing which converts any arbitrary sized data in the form of a fixed size value where the output cannot be converted to input again. But Hash Encoder comes with loss of data and OneHot Encoder increase dimensionality of data. This can be fixed with Binary Encoder. That is the reason I have chosen Binary Encoder to convert my data to binary. I have referred the code from https://analyticsindiamag.com/a-complete-guide-to-categorical-data-encoding/

![image](https://user-images.githubusercontent.com/126722476/224239150-217c428e-5800-45a9-af33-1eab804b8114.png)

## Data Visualization

Data Visualization is the graphical representation of data. It helps in data analysis of large datasets, imbalanced data, recognizing patterns and dependency among the features. Therefore I have plotted barplot from https://seaborn.pydata.org/generated/seaborn.barplot.html and lineplot from https://seaborn.pydata.org/generated/seaborn.lineplot.html to plot the dependencies between features  as shown below.

### Fare vs Pclass vs Survival rate

![image](https://user-images.githubusercontent.com/126722476/224204738-206928a5-ea6f-4066-aae8-99aed5c9a94a.png)

Pclass=1 indicates upper class, Pclass=2 middle and Pclass=3 lower class respectively. So from the above graph, we can observe that the passengers in the upper class have paid more than the remaining two classes. Moreover, when the survived count is considered, it seems that the priority was given to upper class when it comes to saving.

### Dependency of Gender on Survival rate

![image](https://user-images.githubusercontent.com/126722476/224204820-34955be5-3b56-4903-8c89-51fea7580b49.png)

I have used the code from https://www.kaggle.com/code/alexisbcook/titanic-tutorial/notebook to find out the percentage of women and men survived. And I have plotted the features 'Sex' on X-axis and 'Survived' on Y-axs to observe the pattern graphically. It seems that between 74% of the women survived when compared to male its only about 19%. This indicates that priority was given to women.

### Dependency of Embarked on Survival rate

![image](https://user-images.githubusercontent.com/126722476/224204991-f5783253-69de-4d9d-88f8-4200e926c8ef.png)

The Embarked indicates that passengers have embarked the ship from either port C = Cherbourg or Q = Queenstown or S = Southampton. To find out the ports of embarkation from which most of them survived I plotted a line plot between 'Embarked' and 'Survived' columns. It seems that passengers from port C survived more. 

### Family vs Survival rate

![image](https://user-images.githubusercontent.com/126722476/224248368-5f63b0fb-3e67-49b2-a37e-8f49f9b59e3b.png)

I created a new dataframe by combining the columns Parch and SibSp since Parch indicates parents/children and SibSp indiactes Siblings/Spouse. Therefore, I combined them in order as it encloses a family together. I plotted the family count on X-axis and survival rate on y-axis using barplot. We can observe that the passengers who were alone rather than a family. Interesting observation!!!


## Feature Selection

We can build a predictive model by reducing the features which means that all the given features may not depend on target variable, some might be irrevalant and redundant as well. The model can predict the outcome better if we make the model only the necessary and predominant features. We can findout such features by building the HeatMap with their correlation values. These values are both positive and negative. The correlation value between two identical columns is 1. These correlation values indicate the dependency between two features. If the correlation value between a feature and target variable is positive it means that those features are positively correlated (if the correlation value of feature increases, the target variable value also increases). If the correlation value between a feature and target variable is negative it means that those features are negatively correlated (if the correlation value of feature increases, the target variable value decreases).

I found the correlation values using corr() function and have drawn a heatmap referring its seaborn implementation https://seaborn.pydata.org/generated/seaborn.heatmap.html. 


![image](https://user-images.githubusercontent.com/126722476/224252143-adadd20b-3da9-41f6-8253-c4ccc5dbbe84.png)
In the above code, parameter annot=True is used to insert correlation values in the heatmap grid and I have set the size to 8 with paramter annot_kws. 

By observing the correlation values from above Heatmap, I summarized that the range is not that great. The absolute highest value is Sex followed by Pclass and Fare. Here Sex and Pclass are negatively correlated with Survived which means passengers in upper class were given priority than lower,middle classes and Fare is positively correlated. From the heatmap, we can see that Parch and SibSp are highly correlated that makes sense as if we both combine together it gives the family data. Furthermore, Pclass and fare are also highly negativey correlated. So to see this pictorially, I have plotted a graph between them as below

![image](https://user-images.githubusercontent.com/126722476/224254718-1cdc1326-b749-423e-a27b-2d0196ce1b64.png)

So when the Pclass=1 the Fare is more and the fare has been decreasing for the passengers in Pclass=3 which makes sense as the rich people were given priority.

I have found the correlation coefficients with the target variable Survived referring code from https://datascience.stackexchange.com/questions/39137/how-can-i-check-the-correlation-between-features-and-target-variable and modified input accordingly.

**correlation values with target variable**

![image](https://user-images.githubusercontent.com/126722476/224257454-70ae2d66-1dc8-45e0-8eef-b90049e0a06f.png)

After analysing the correlation values from heat map, I have referred code snippet from https://towardsdatascience.com/feature-selection-in-python-using-filter-method-7ae5cbc4ee05 in selecting abs threshold value. I chose threshold to be abs(0.08) and selected only features having correlation values above abs(0.08). The features which were selected are Pclass, Sex, Parch, Ticket, Fare, Cabin, Embarked.

![image](https://user-images.githubusercontent.com/126722476/224258049-304faf16-89e4-4b41-89d9-de0dfd3e24a3.png)












