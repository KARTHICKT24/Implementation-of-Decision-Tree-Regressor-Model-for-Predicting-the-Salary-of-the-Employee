# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packagesprint the present data

2.print the present data

3.print the null value

4.using decisiontreeRegressor, find the predicted values,mse,r2

5.print the result 

## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: KARTHICK KISHORE T
RegisterNumber:  212223220042
```
```

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

data = pd.read_csv("Salary.csv")
print(data.head(), "\n")
print(data.info(), "\n")
print(data.isnull().sum(), "\n")

le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
print(data.head(), "\n")

x = data[["Position", "Level"]]
y = data["Salary"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

dt = DecisionTreeRegressor(random_state=2)
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print(mse)
print(r2)

print(dt.predict([[5, 6]])[0])
```
## Output:
**
<img width="130" height="168" alt="image" src="https://github.com/user-attachments/assets/93e66738-b097-40ae-ac76-b1f0fb4f9bdb" />
**
<img width="230" height="185" alt="image" src="https://github.com/user-attachments/assets/7f6c28c5-0428-4790-94c8-783fd826ac78" />
**
<img width="113" height="223" alt="image" src="https://github.com/user-attachments/assets/8799e26c-89b4-4172-b05d-38659498ca5c" />
**
<img width="158" height="40" alt="image" src="https://github.com/user-attachments/assets/044c22d4-45ff-4d75-81a9-cff0c8a872eb" />
**
<img width="157" height="19" alt="image" src="https://github.com/user-attachments/assets/ea3748f5-d3e3-419d-bbd4-665cdc0f543e" />
**
## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
