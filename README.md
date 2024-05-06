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
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: gokul sachin .k
RegisterNumber:  212223220025

import pandas as pd
data=pd.read_csv("/content/Salary_EX7.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
x_train
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt
clf=DecisionTreeRegressor(criterion='gini')
plt.figure(figsize=(20,8))
plot_tree(dt,feature_names=x.columns,filled=True)
plt.show()

```

## Output:
![169694235-41a469cc-ff3e-4c56-b36c-029319ef1f94](https://github.com/vksachin2018/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/149347526/d22bf4c0-6e8f-42e4-9f30-f1d5b7be0df6)
![169694238-85077655-4a64-4334-b451-997c7ea1937d](https://github.com/vksachin2018/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/149347526/870b9d93-fdc6-4318-884a-8569a583610e)
![169694242-dd7cae7b-50db-4864-96aa-ca8eb07514e3](https://github.com/vksachin2018/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/149347526/ff63501b-8c67-44a6-8c8a-caa275bbd3c0)
![169694248-eefed989-8fc7-4e80-b3af-992667d1936a](https://github.com/vksachin2018/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/149347526/b684a3c1-ae48-4c76-8a3d-7a95c43edb3d)
![169694252-b17fc5dd-22fd-46e0-b8de-991fd12528ed](https://github.com/vksachin2018/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/149347526/03ee2f6e-992a-4b72-b8bb-66fedd693110)
![169694252-b17fc5dd-22fd-46e0-b8de-991fd12528ed](https://github.com/vksachin2018/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/149347526/f79f9c17-28cc-43ed-a04b-36570404e6ae)
![169694255-16669af0-0ed0-416e-b387-d63f2f3e9dc3](https://github.com/vksachin2018/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/149347526/0dce8deb-25c8-4bdb-9da5-99db4319cc4e)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
