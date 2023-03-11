# Titanic- Data Analysis, Visualization and Accuracy score calculation
![image](https://user-images.githubusercontent.com/126722476/224194133-9622c3aa-507b-4866-8884-d755e7fa4f98.png)

**Source code available here** : https://github.com/Moukthika1253/moukthika_dasika/blob/main/titanic_classification.ipynb

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

I found the missing values from both train and test data set and summed them up referring the code from ***[1]***

### Before fixing missing values

|Missing values in training_data|missing values in testing_data|
|---|---|
|![image](https://user-images.githubusercontent.com/126722476/224204551-8bb670f6-b722-43da-9112-8d9f5034d671.png)|![image](https://user-images.githubusercontent.com/126722476/224204593-cdfb0a69-2013-4438-8c95-dd5cac6c5ec5.png)|

From the above table we can tell that in training data columns Age, Cabin and Embarked have missing values. In test data, Age, Fare and Cabin have missing values.

### After fixing missing values
One of the better ways to deal with missing data is to fill them with their mean/median if the data is numerical and mode if the data is categorical. Since we have missing values in both categorical and numerical data I have filled them with the Mode(most repeating value) in Cabin and Embarked columns, with Mean(average) in Age, Fare columns. .

```python
training_data['Age']=training_data['Age'].fillna(training_data['Age'].mean())
training_data['Cabin']=training_data['Cabin'].fillna(training_data['Cabin'].mode()[0])
training_data['Embarked']=training_data['Embarked'].fillna(training_data['Embarked'].mode()[0])
testing_data['Age']=testing_data['Age'].fillna(testing_data['Age'].mean())
testing_data['Fare']=testing_data['Fare'].fillna(testing_data['Fare'].mean())
testing_data['Cabin']=training_data['Cabin'].fillna(testing_data['Cabin'].mode()[0])
```


Referred above code from ***[2]***



|training_data|testing_data|
|---|---|
|![image](https://user-images.githubusercontent.com/126722476/224204652-ab8a8fd9-0578-4a3b-8c1a-dd11a73235eb.png)|![image](https://user-images.githubusercontent.com/126722476/224204684-7937212b-c57a-46cf-8335-3f57adf216af.png)|


## Data Encoding - Binary Encoder

Data Encoding is one of the pre-processing techniques. The encoding process involves converting the categorical data into numerical data. This is essential since majority of the algorithms need the data to be numerical and it also helps in improving the performance of the learning model as it can interpret the relationship between features and target variable in a better way. Therefore, I converted the categorical data (Name, Sex, Cabin, Ticket, Embarked) to numerical data in both training and test datasets using category_encoders library by referring code from ***[3]***

Sample code

```python
training_data['Name'] =training_data['Name'].astype('category').cat.codes
testing_data['Name'] =testing_data['Name'].astype('category').cat.codes
```


Binary encoder is a combination of OneHot Encoder and Hash Encoder. In OneHot Encoder the categorical data in nominal form is converted to binary values by creating new dummy variables. The Hash Encoder does the same but encodes them using hashing which converts any arbitrary sized data in the form of a fixed size value where the output cannot be converted to input again. But Hash Encoder comes with loss of data and OneHot Encoder increase dimensionality of data. This can be fixed with Binary Encoder. That is the reason I have chosen Binary Encoder to convert my data to binary. I have referred the code from ***[4]***

```python
encoder=c.BinaryEncoder(cols=['Name','Sex','Ticket','Cabin','Embarked'],return_df=True)
encoder.fit_transform(training_data)
encoder.fit_transform(testing_data)
```

## Data Visualization

Data Visualization is the graphical representation of data. It helps in data analysis of large datasets, imbalanced data, recognizing patterns and dependency among the features. Therefore I have plotted barplot from ***[5]***  and lineplot from ***[6]*** , ***[7]*** to plot the dependencies between features  as shown below.

### Fare vs Pclass vs Survival rate

![image](https://user-images.githubusercontent.com/126722476/224204738-206928a5-ea6f-4066-aae8-99aed5c9a94a.png)

Pclass=1 indicates upper class, Pclass=2 middle and Pclass=3 lower class respectively. So from the above graph, we can observe that the passengers in the upper class have paid more than the remaining two classes. Moreover, when the survived count is considered, it seems that the priority was given to upper class when it comes to saving.

### Dependency of Gender on Survival rate

![image](https://user-images.githubusercontent.com/126722476/224204820-34955be5-3b56-4903-8c89-51fea7580b49.png)

I have used the code from ***[1]*** to find out the percentage of women and men survived. And I have plotted the features 'Sex' on X-axis and 'Survived' on Y-axs to observe the pattern graphically. It seems that between 74% of the women survived when compared to male its only about 19%. This indicates that priority was given to women.

### Dependency of Embarked on Survival rate

![image](https://user-images.githubusercontent.com/126722476/224204991-f5783253-69de-4d9d-88f8-4200e926c8ef.png)

The Embarked indicates that passengers have embarked the ship from either port C = Cherbourg or Q = Queenstown or S = Southampton. To find out the ports of embarkation from which most of them survived I plotted a line plot between 'Embarked' and 'Survived' columns. It seems that passengers from port C survived more. 

### Family vs Survival rate

![image](https://user-images.githubusercontent.com/126722476/224248368-5f63b0fb-3e67-49b2-a37e-8f49f9b59e3b.png)

I created a new dataframe by combining the columns Parch and SibSp since Parch indicates parents/children and SibSp indiactes Siblings/Spouse. Therefore, I combined them in order as it encloses a family together. I plotted the family count on X-axis and survival rate on y-axis using barplot. We can observe that the passengers who were alone rather than a family. Interesting observation!!!


## Feature Selection

We can build a predictive model by reducing the features which means that all the given features may not depend on target variable, some might be irrevalant and redundant as well. The model can predict the outcome better if we make the model only the necessary and predominant features. We can findout such features by building the HeatMap with their correlation values. These values are both positive and negative. The correlation value between two identical columns is 1. These correlation values indicate the dependency between two features. If the correlation value between a feature and target variable is positive it means that those features are positively correlated (if the correlation value of feature increases, the target variable value also increases). If the correlation value between a feature and target variable is negative it means that those features are negatively correlated (if the correlation value of feature increases, the target variable value decreases).

I found the correlation values using corr() function and have drawn a heatmap referring its seaborn implementation ***[8]***


```python
sns.heatmap(training_data.corr(), annot=True,annot_kws={'size': 8})
```
In the above code, parameter annot=True is used to insert correlation values in the heatmap grid and I have set the size to 8 with paramter annot_kws. 

By observing the correlation values from above Heatmap, I summarized that the range is not that great. The absolute highest value is Sex followed by Pclass and Fare. Here Sex and Pclass are negatively correlated with Survived which means passengers in upper class were given priority than lower,middle classes and Fare is positively correlated. From the heatmap, we can see that Parch and SibSp are highly correlated that makes sense as if we both combine together it gives the family data. Furthermore, Pclass and fare are also highly negativey correlated. So to see this pictorially, I have plotted a graph between them as below

![image](https://user-images.githubusercontent.com/126722476/224254718-1cdc1326-b749-423e-a27b-2d0196ce1b64.png)

So when the Pclass=1 the Fare is more and the fare has been decreasing for the passengers in Pclass=3 which makes sense as the rich people were given priority.

I have found the correlation coefficients with the target variable Survived referring code from ***[9]*** and modified input accordingly.

**correlation values with target variable**

```python
col_names=['PassengerId','Survived','Pclass',"Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]
corr_values=training_data[training_data.columns[0:]].corr()['Survived']
print(corr_values)
```

![image](https://user-images.githubusercontent.com/126722476/224257454-70ae2d66-1dc8-45e0-8eef-b90049e0a06f.png)

After analysing the correlation values from heat map, I have referred code snippet from ***[10]*** in selecting abs threshold value. I chose threshold to be abs(0.08) and selected only features having correlation values above abs(0.08). The features which were selected are Pclass, Sex, Parch, Ticket, Fare, Cabin, Embarked.

## Random Forest - Learning model, Prediction, Accuracy based on features selected from correlation values

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
y = training_data["Survived"]
features = ["Pclass","Sex","Parch","Ticket","Fare","Cabin","Embarked"]
X = pd.get_dummies(training_data[features])
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.37, random_state=42)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions_c = model.predict(X_test)
print(metrics.accuracy_score(y_test,predictions_c))
```

I have used the code given by ***[1]*** for training the Random Forest model and predicting with the test data. I have given the features which were selected from above analysis as the X variable and survived as the target y variable. I have split the training data into 63%-train, 37%- test data referring the code from ***[11]***. 
Then I have predicted on the test data from the split and calculated the accuracy score using the metrics function from ***[12]***
I got 0.833 as accuracy. Next I have calculated predictions on testing data using above features. After submitting the output csv to the competetion I got the accuracy as below

![image](https://user-images.githubusercontent.com/126722476/224372094-b95484db-b852-4bdd-9f63-de370ee16639.png)

This is when I observed that its leading to overfitting because the test accuracy is less than my train accuracy. So I have split the training data into train=67% instead of 63% and test from 37% to 33%.

I wanted to improve the accuracy score. So I implemented another feature selection method which is Chi-Square test referred from ***[13]***

```python
from sklearn.feature_selection import chi2
X = training_data.drop('Survived',axis=1)
y = training_data['Survived']
chi_scores = chi2(X,y)
chi_scores
```

![image](https://user-images.githubusercontent.com/126722476/224380941-b594f91a-2190-4bb8-b214-48ebe949d27c.png)

```python
p_values = pd.Series(chi_scores[1],index = X.columns)
p_values.sort_values(ascending = False , inplace = True)
p_values.plot.bar()
```

![image](https://user-images.githubusercontent.com/126722476/224381171-f6345960-8740-42a8-974c-5f501f55a874.png)

According to the Sampath kumar, from ***[13]***, the features SibSp and PassengerId have high p-value which indicates that they are independent from the target variable and they need not be considered for training model. Hence I selected "Pclass","Name","Sex","Age","Parch","Ticket","Fare","Cabin","Embarked" as the features to train my model using various classifiers.

I have split the data again into train (67%) and test (33%) with the features selected as below. I create a list called accuracy, which will append the accuracies calculated with various classifiers.

```python
y = training_data["Survived"]
features = ["Pclass","Name","Sex","Age","Parch","Ticket","Fare","Cabin","Embarked",]
X = pd.get_dummies(training_data[features])
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.37, random_state=42)
accuracy=[]
```

## Model prediction using various classifiers

Code for Decision Tree, Logisitic Regression was referred from ***[14]***

### Linear SVM

SVC stands for Support vector Machine Classifier, it is called linear SVC because in python this algorithm gives us the best fit hyperplane which differentiates or categorizes different features in the data. In this algorithm, we will calculate the vectore which optimizes the line and to ensure that the closes point in each group lies farsthest from each other in that group. I chose the kernel to be linear so that the algorithm differentiates features using a line and the value C indicates how perfectly we want to fit the data so 1.0 is usually considered as the best default parameter. I referred code from ***[16]***, modified nputs according to my requirement as below. 

```python
from sklearn import svm
linear_svm = svm.SVC(kernel='linear', C = 1.0)
linear_svm.fit(X,y)
predictions_svm=linear_svm.predict(X_test)
accuracy.append(metrics.accuracy_score(y_test,predictions_svm))
```

### Decision Tree classifier

Decision tree classifier is a supervised machine learning algorithm as it learns the data using its labels. It woeks on both continous dependent and categorical variables. The algorithm considers an instance compares,traverses through a tree internally,selecting important features with a determined conditional statement.

```python
from sklearn.tree import DecisionTreeClassifier
dc=DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=3, min_samples_leaf=5)
dc.fit(X,y)
predictions_dc=dc.predict(X_test)
accuracy.append(metrics.accuracy_score(y_test,predictions_dc))
```

### Logistic Regression classifier

This classifier is a supervised machine learning algorithm. It is used for predicting discrete values. A logistic function is used to predict the probability of an event whose outcome is between 0 and 1. I have set maximum iterations to 1000 since I got an error "TOTAL NO. of ITERATIONS REACHED LIMIT". Then I trained the model and appended accuracy score to the list.

**training the model**

```python
from sklearn import linear_model
lr_model= linear_model.LogisticRegression(max_iter=1000)  
lr_model.fit(X, y)  
predictions_lr = lr_model.predict(X_test)
accuracy.append(metrics.accuracy_score(y_test,predictions_lr))
```

## Random Forest Classifier

This classifier fits a number of decision tree classifiers on various features of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. I used the Kaggle code to train my model with random forest classifier and then calculated test data predictions. Apended the accuracy score in the end. I have created a dataframe 'class_accuracy' by creating two columns one for the classifier names and other their accuracies. Then using the seaborn I plotted a line graph with classifier names on x-axis and accuracy scores on y-axis. From the graph I observed that highest accuracy score was achieved by Random Forest classifier. Therefore I calculated the test predictions from this classifier and submitted the output file. I submitted the output file multiple times with random forest classifier predictions. The highest accuracy according to Kaggle leaderboard I got so far is 0.78468.

**train the model**

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions_rf = model.predict(X_test)
accuracy.append(metrics.accuracy_score(y_test,predictions_rf))
```

**test the model**

```python
test_predict = model.predict(testing_data[features])
test_predict
```

**Accuracy plotting**

```python
class_accuracy=[["Linear SVM",accuracy[0]],["Decision Tree",accuracy[1]],["Logisitic Regression",accuracy[2]],["Random Forest",accuracy[3]]]
df=pd.DataFrame(class_accuracy,columns=["Classifiers","Accuracies"])
print(df)
sns.lineplot(data=df,x=df["Classifiers"],y=df["Accuracies"])
```

![image](https://user-images.githubusercontent.com/126722476/224450720-cceb530d-04c5-4aa3-b5f3-fafa091ecd81.png)

![image](https://user-images.githubusercontent.com/126722476/224450738-5c804b34-4e3b-48b3-b547-3734060c088c.png)

![image](https://user-images.githubusercontent.com/126722476/224450765-d2b0b9b5-79bb-427a-a453-1fffc6c316b7.png)

## Challenges

## Contribution

## References

**SNo**|**URL**
 ---|---|
|[1]|[https://practicaldatascience.co.uk/data-science/how-to-use-isna-to-check-for-missing-values-in-pandas-dataframes]|

|[2]|[https://vitalflux.com/pandas-impute-missing-values-mean-median-mode/]|

|[3]|[https://pbpython.com/categorical-encoding.html]|

|[4]|[https://analyticsindiamag.com/a-complete-guide-to-categorical-data-encoding/]|

|[5]|[https://seaborn.pydata.org/generated/seaborn.barplot.html]|

|[6]|[https://seaborn.pydata.org/generated/seaborn.lineplot.html]|

|[7]|[https://seaborn.pydata.org/generated/seaborn.countplot.html]|

|[8]|[https://seaborn.pydata.org/generated/seaborn.heatmap.html]|

|[9]|[https://datascience.stackexchange.com/questions/39137/how-can-i-check-the-correlation-between-features-and-target-variable
]|

|[10]|[ https://towardsdatascience.com/feature-selection-in-python-using-filter-method-7ae5cbc4ee05]|

|[11]|[https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)]|

|[12]|[https://scikitlearn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html]|

|[13]|[https://towardsdatascience.com/chi-square-test-for-feature-selection-in-machine-learning-206b1f0b8223]|

|[14]|[https://data-flair.training/blogs/machine-learning-algorithms-in-python/]|




