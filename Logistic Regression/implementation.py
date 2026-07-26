import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import matplotlib.pyplot as plt

# --------------------------------
# 1. Dataset Load
# --------------------------------

df = pd.read_csv("driving_exam_scores.csv")
# print(df.head())
# print(df.info())


# --------------------------------
# 2. Target ko Numeric mein Convert
# --------------------------------

df["success_status"] = df["success_status"].map({
    "Yes": 1,
    "No": 0
})

# --------------------------------
# 3. Input Features
# --------------------------------

X = df[
    [
        "age",
        "total_study_hours",
        "practice_exams_taken",
        "average_practice_score",
        "exam_attempt_number"
    ]
]

# --------------------------------
# 4. Target
# --------------------------------

y = df["success_status"]


# --------------------------------
# 5. Train-Test Split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------
# 6. Model Banana
# --------------------------------

model = LogisticRegression()


# --------------------------------
# 7. Model Train Karna
# --------------------------------

model.fit(X_train, y_train)


# --------------------------------
# 8. Prediction
# --------------------------------

y_pred = model.predict(X_test)


# --------------------------------
# 9. Accuracy
# --------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")

# --------------------------------
# 10. User Input
# --------------------------------

age = float(input("Enter Age: "))

study_hours = float(
    input("Enter Total Study Hours: ")
)

practice_exams = int(
    input("Enter Number of Practice Exams Taken: ")
)

practice_score = float(
    input("Enter Average Practice Score: ")
)

attempt_number = int(
    input("Enter Exam Attempt Number: ")
)


# --------------------------------
# 11. User Input ko DataFrame mein Convert
# --------------------------------

new_data = pd.DataFrame({
    "age": [age],
    "total_study_hours": [study_hours],
    "practice_exams_taken": [practice_exams],
    "average_practice_score": [practice_score],
    "exam_attempt_number": [attempt_number]
})


# --------------------------------
# 12. Prediction
# --------------------------------

prediction = model.predict(new_data)


# --------------------------------
# 13. Probability
# --------------------------------

probability = model.predict_proba(new_data)


# --------------------------------
# 14. Result
# --------------------------------

if prediction[0] == 1:
    print("\nPrediction: Successful")
else:
    print("\nPrediction: Not Successful")


# --------------------------------
# 15. Probability Display
# --------------------------------

print("Success Probability:",round(probability[0][1] * 100, 2),"%")

print("Failure Probability:",round(probability[0][0] * 100, 2),"%")

#----------------------------
# 16. Visualization
#----------------------------

# status_count = df["success_status"].map({
#     1:"Successful",
#     0:"Not Successful"
# }).value_counts()


#BAR CHART
# plt.bar(
#     status_count.index,
#     status_count.values
#         )

# plt.xlabel("Exam Result")
# plt.ylabel("No of candidates")
# plt.title("Driving Exam Success Status")

# plt.show()

# #PIE CHART
# plt.pie(
#     status_count.values,
#     labels=status_count.index,
#     autopct="%1.1f%%"
# )

# plt.title("Driving Exam Success Distribution")
# plt.show()

# #STUDY HOURS VS FINAL EXAM SCORE
# plt.scatter(
#     df["total_study_hours"],
#     df["final_exam_score"]
# )

# plt.xlabel("Total study hours")
# plt.ylabel("Final exam score")
# plt.title("Study Hours vs Final Exam Score")
# plt.show()
