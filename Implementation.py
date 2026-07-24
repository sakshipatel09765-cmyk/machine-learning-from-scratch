import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

data = pd.read_csv("Student.csv")

X = data[['Hours']]     #double brackets -> 2d=D input
Y = data['Score']       #Target column

model = LinearRegression()
model.fit(X, Y)
predicted_score = model.predict(X)

#evaluate
mae = mean_absolute_error(Y, predicted_score)
mse = mean_squared_error(Y, predicted_score)
rmse = np.sqrt(mse)

#show results
print("Mean Absolute error (MAE):", mae)
print("Mean sqared error (MSE):", mse)
print("Root Mean sqared error (RMSE):", rmse)


