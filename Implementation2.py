import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

#load dataset
data = pd.read_csv("student_exam_data_new.csv")

#Input and output
x = data[['Study Hours']]
y = data['Pass/Fail']

#train model
model = LogisticRegression()
model.fit(x, y)

predicted_Score = model.predict(x)

#valid regression metrix
#evaluate
mae = mean_absolute_error(y, predicted_Score)
mse = mean_squared_error(y, predicted_Score)
rmse = np.sqrt(mse)
r2 = r2_score(y, predicted_Score)

#show results
print("Mean Absolute error (MAE): ", round(mae, 2))
print("Mean sqared error (MSE): ", round(mse, 2))
print("Root Mean sqared error (RMSE): ", round(rmse, 2))
print("R^2 Score (Model Accuracy): ", round(r2, 4))   #closer to 1 = better

# #histogram
# plt.figure(figsize=(10, 6))
# plt.hist(data["Pass/Fail"], bins=30, color='skyblue', edgecolor='black')
# plt.title("Distribution of Final Exam Result")
# plt.xlabel("Exam Result")
# plt.ylabel("Number of students")
# plt.grid(True)
# plt.show()

# Count Pass and Fail students
result_counts = data["Pass/Fail"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(["Fail", "Pass"], result_counts.values, edgecolor="black")

plt.title("Distribution of Pass and Fail Students")
plt.xlabel("Exam Result")
plt.ylabel("Number of Students")
plt.grid(axis="y")

plt.show()
