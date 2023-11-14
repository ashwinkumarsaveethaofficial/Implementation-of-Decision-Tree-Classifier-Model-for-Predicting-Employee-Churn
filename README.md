## Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.
   
   
## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: S Ashwinkumar
RegisterNumber:  212222040020

import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test, y_pred)
accuracy

dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])

```

## Output:
# data.head():
![Screenshot 2023-10-05 093327](https://github.com/arun1111j/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128461833/50a12d23-1ec9-4363-9ec6-e0e37d12df5b)
# data.info():
![Screenshot 2023-10-05 093419](https://github.com/arun1111j/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128461833/ea54c164-e273-4791-b350-dbfd0b6ecf45)
# isnull() and sum():
![Screenshot 2023-10-05 093443](https://github.com/arun1111j/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128461833/9920c4e1-b3ae-40d8-acf1-f1456451a051)
# data value counts():
![Screenshot 2023-10-05 093510](https://github.com/arun1111j/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128461833/8373a3ed-86cf-4a35-9fbb-d897d1d26270)
# data.head() for salary:
![Screenshot 2023-10-05 093545](https://github.com/arun1111j/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128461833/64da3b10-d1d2-4c94-be7d-3f827560e59d)
# x.head():
![Screenshot 2023-10-05 093615](https://github.com/arun1111j/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128461833/4ae2b923-89a5-49b5-95fc-0a368f6cca97)
# accuracy value:
![Screenshot 2023-10-05 093644](https://github.com/arun1111j/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128461833/21dbd1bf-afba-44f9-a3ad-5ec1eeb8a358)
# data prediction:
![Screenshot 2023-10-05 093717](https://github.com/arun1111j/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128461833/9b393fae-2a4b-4ab6-b77d-d34b7a948f2d)




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
