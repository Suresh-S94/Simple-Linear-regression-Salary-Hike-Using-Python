import numpy as np # linear algebra
import matplotlib.pyplot as plt # Plotting
import pandas as pd

#importing the data set
dataset = pd.read_csv('E:\\Data Science_Excelr\\Simple Linear Regression\\Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values
X = X.reshape(-1,1)
y = y.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred=regressor.predict(X_test)

#Visualsing the training set results
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),)
plt.title('Salary vs Experience(Train set)')
plt.xlabel('Experience in years')
plt.ylabel('Salary')

#Visualising the test set results
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs experience(Test set)')
plt.xlabel('Experience in years')
plt.ylabel('Salary')
plt.show()

