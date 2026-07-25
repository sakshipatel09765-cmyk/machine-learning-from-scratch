import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#data set load karna
df = pd.read_csv("salaries.csv")

#Input Feature
x = df[["years_of_experience"]]

#Target
y = df["salary"]

#Train-Test the model
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=True
)

#model banana
model = LinearRegression()

#model train karna 
model.fit(x_train,y_train)

#User se input lena
yoe = float(input("Enter Year Of Experience: "))

#user input ko data frame mein convert karna
new_data = pd.DataFrame({
    "years_of_experience": [yoe]
})

#prediction
prediction = model.predict(new_data)

#result
print("Predicted Salary: ", round(prediction[0],2))


#MAE, MSE, RMSE ,r^2 score
y_predict = model.predict(x_test)

'''
here y_test is actual ans and y_predict is predicted ans
'''
mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_predict)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# -----------------------------------
# Visualization
# -----------------------------------

# Actual data points
plt.scatter(
    x,
    y,
    label="Actual Data"
)

# Linear Regression predictions
y_line = model.predict(x)

# Regression line
plt.plot(
    x,
    y_line,
    label="Linear Regression Line"
)

# Labels
plt.xlabel("Years of Experience")
plt.ylabel("Salary")

# Title
plt.title("Years of Experience vs Salary")

# Show legend
plt.legend()

# Display graph
plt.show()

