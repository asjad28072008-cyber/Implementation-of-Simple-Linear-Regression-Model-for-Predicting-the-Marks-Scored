# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Muhammad Asjad  E
RegisterNumber:  25013957
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/student_scores.csv')
df.head()
df.tail()
X=df.iloc[:,:-1].values
print(*X)
Y=df.iloc[:,1].values
print(*Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
#Y_pred
print(*Y_pred)
#Y_test
print(*Y_test)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test, Y_test, color="blue")
plt.plot(X_test, regressor.predict(X_test), color="green")
plt.title('Testing set (Hours vs Scores)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
```

## Output:
To Read Head and Tail Files
    


<h3>Hours<img width="521" height="222" alt="Screenshot 2025-10-04 204230" src="https://github.com/user-attachments/assets/59726817-3443-43ff-8ab1-8ed92ae47227" />Scores</h3>



<h3>Hours<img width="543" height="216" alt="Screenshot 2025-10-04 204259" src="https://github.com/user-attachments/assets/cd465ca3-b903-43ab-b712-b3a1bba6cea3" />Scores</h3>

Compare Dataset

Hours:

[2.5] [5.1] [3.2] [8.5] [3.5] [1.5] [9.2] [5.5] [8.3] [2.7] [7.7] [5.9] [4.5] [3.3] [1.1] [8.9] [2.5] [1.9] [6.1] [7.4] [2.7] [4.8] [3.8] [6.9] [7.8]


Scores:


21 47 27 75 30 20 88 60 81 25 85 62 41 42 17 95 30 24 67 69 30 54 35 76 86


Predicted Value

17.042891792891766 33.51695376695375 74.21757746757748 26.73351648351646 59.68164043164043 39.33132858132856 20.91914166914164 78.09382734382734 69.37226512226513
20 27 69 30 62 35 24 86 76


Graph For Training Set

<img width="662" height="500" alt="Screenshot 2025-10-04 205216" src="https://github.com/user-attachments/assets/8fc68ba0-fcdf-44d6-a623-27f028025e83" />


Graph For Testing Set


<img width="704" height="514" alt="Screenshot 2025-10-04 205903" src="https://github.com/user-attachments/assets/7e6fcd54-6611-46d8-b9c8-8516d2d5ddca" />


Error

<img width="456" height="80" alt="Screenshot 2025-10-04 205915" src="https://github.com/user-attachments/assets/8ad8c6ff-786d-4add-b685-d9274f8eafc0" />




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
