# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the necessary packages using import statement.
2. Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().
3. Import KMeans and use for loop to cluster the data.
4. Predict the cluster and plot data graphs.
5. Print the outputs and end the program

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: S Ashwinkumar
RegisterNumber:  212222040020

import pandas as pd
import numpy as np
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
data["salary"]=l.fit_transform(data["salary"])
data.head()
data["Departments "]=l.fit_transform(data["Departments "])
data.head()
data.info()
data.shape
x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','Departments ','salary']]
x.head()
x.shape
x.info()
y=data['left']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print(y_pred)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("Accuracy = ",accuracy)
dt.predict([[0.5,0.8,9,260,6,0,1,2,1]])
*/
```

## Output:

### Read csv file:

![image](https://github.com/shalinikannan23/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118656529/60fbf767-4631-44d5-9926-2c24d3b628e2)

### Dataset info:

![image](https://github.com/shalinikannan23/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118656529/c883f44b-7db1-4085-bd21-7fe0e5cf985d)

![image](https://github.com/shalinikannan23/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118656529/818d29c1-0319-4ad5-85bf-42e6b8ad4488)

### Dataset Value count:

![image](https://github.com/shalinikannan23/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118656529/79d18e09-70f9-470f-90d4-53c6a825af93)

### Dataset head:

![image](https://github.com/shalinikannan23/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118656529/65a4703e-2083-45ec-81f4-463a1a302574)

### Data info:

![image](https://github.com/shalinikannan23/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118656529/0374428c-44d8-4a2c-a72a-6b3f2d820604)

### Dataset shape:

![image](https://github.com/shalinikannan23/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118656529/24970a9c-945c-4e0f-aeb2-fbdd4f82a8ee)

![image](https://github.com/shalinikannan23/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118656529/105592b3-78be-48dc-a29a-65f26ce957e5)

### Y-Pred:

![image](https://github.com/shalinikannan23/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118656529/417f3652-06e2-4fa6-9d79-16dd29a2cec1)

### Accuracy:

![image](https://github.com/shalinikannan23/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118656529/bedddd9e-cb40-4b59-9257-26e0afc28ae7)

### Dataset Predict:

![image](https://github.com/shalinikannan23/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118656529/a6d636e5-3a36-4016-b04b-b7a38512cae0)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
