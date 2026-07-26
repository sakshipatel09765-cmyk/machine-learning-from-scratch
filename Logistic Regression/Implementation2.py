import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

#load dataset
data = pd.read_csv("student_exam_data_new.csv")

#Input and output
x = data[['Study Hours']]
y = data['Pass/Fail']

X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

#train model
model = LogisticRegression()
model.fit(X_train, y_train)

#Accuray of model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy,4))
print("Model Accuracy (%): ", round(accuracy *100,2),"%")

#User input

hour = float(input("Enter no. of hours you study: "))

new_data = pd.DataFrame({
    "Study Hours": [hour]
})
predicted_Score = model.predict(new_data)

if predicted_Score[0] == 1:
    print("Predicted Result: Pass")
else:
    print("Predicted Result: Fail")

# #histogram
# plt.figure(figsize=(10, 6))
# plt.hist(data["Pass/Fail"], bins=30, color='skyblue', edgecolor='black')
# plt.title("Distribution of Final Exam Result")
# plt.xlabel("Exam Result")
# plt.ylabel("Number of students")
# plt.grid(True)
# plt.show()

# Count Pass and Fail students
# result_counts = data["Pass/Fail"].value_counts().sort_index()
avg_hour = data.groupby("Pass/Fail")["Study Hours"].mean()

plt.figure(figsize=(8, 5))
plt.bar(
    range(len(avg_hour)),
    avg_hour.values,
    edgecolor = "black"
    )
plt.xticks([0,1]),["Fail,Pass"]

plt.title("Average Study Hours of Pass and Fail Students")
plt.xlabel("Exam Result")
plt.ylabel("Average Study Hours")

plt.grid(axis="y")

plt.show()
