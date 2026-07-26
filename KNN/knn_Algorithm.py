import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# 1. LOAD THE DATASET
# =========================================================

df = pd.read_csv("KNNAlgorithmDataset.csv")

# print("First 5 Rows:")
# print(df.head())

# print("\nDataset Shape:")
# print(df.shape)


# =========================================================
# 2. CHECK MISSING VALUES
# =========================================================

# print("\nMissing Values:")
# print(df.isnull().sum())


# =========================================================
# 3. DROP UNNECESSARY COLUMNS
# =========================================================

df = df.drop(['id', 'Unnamed: 32'],axis=1)


# =========================================================
# 4. SEPARATE FEATURES AND TARGET
# =========================================================

# x = 30 medical features
# y = diagnosis

x = df.drop(
    'diagnosis',
    axis=1
)

y = df['diagnosis']


# =========================================================
# 5. CONVERT TARGET INTO NUMBERS
# =========================================================

# M = 1 → Malignant
# B = 0 → Benign

y = y.map({
    'M': 1,
    'B': 0
})


# =========================================================
# 6. TRAIN-TEST SPLIT
# =========================================================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================================
# 7. FEATURE SCALING
# =========================================================

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)


# =========================================================
# 8. CREATE FINAL KNN MODEL
# =========================================================

knn = KNeighborsClassifier(
    n_neighbors=5
)


# =========================================================
# 9. TRAIN THE MODEL
# =========================================================

knn.fit(
    x_train,
    y_train
)


# =========================================================
# 10. MAKE PREDICTIONS
# =========================================================

y_pred = knn.predict(
    x_test
)


# =========================================================
# 11. CHECK ACCURACY
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n================================")
print("FINAL MODEL RESULTS")
print("================================")

print("\nAccuracy:", accuracy)

print(
    "Accuracy Percentage:",
    accuracy * 100,
    "%"
)


# =========================================================
# 12. CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# =========================================================
# 13. CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:")

print(classification_report(y_test,y_pred))


# # =========================================================
# # 14. CONFUSION MATRIX VISUALIZATION
# # =========================================================

# plt.figure(
#     figsize=(6, 5)
# )

# sns.heatmap(
#     cm,
#     annot=True,
#     fmt='d',
#     cmap='Blues'
# )

# plt.xlabel(
#     "Predicted"
# )

# plt.ylabel(
#     "Actual"
# )

# plt.title(
#     "KNN Confusion Matrix"
# )

# plt.show()


# =========================================================
# 15. FIND ACCURACY FOR K = 1 TO 20
# =========================================================

k_values = range(
    1,
    21
)

accuracy_values = []


for k in k_values:

    # Create temporary KNN model
    knn_temp = KNeighborsClassifier(
        n_neighbors=k
    )

    # Train temporary model
    knn_temp.fit(
        x_train,
        y_train
    )

    # Make predictions
    y_pred_temp = knn_temp.predict(
        x_test
    )

    # Calculate accuracy
    accuracy_temp = accuracy_score(
        y_test,
        y_pred_temp
    )

    # Store accuracy
    accuracy_values.append(
        accuracy_temp
    )


# =========================================================
# 16. PRINT ACCURACY FOR EACH K
# =========================================================

print("\n================================")
print("ACCURACY FOR DIFFERENT K VALUES")
print("================================")

for k, accuracy_value in zip(
    k_values,
    accuracy_values
):

    print(
        "K =",
        k,
        "Accuracy =",
        accuracy_value
    )


# =========================================================
# 17. FIND BEST K
# =========================================================

best_k = k_values[
    accuracy_values.index(
        max(accuracy_values)
    )
]

best_accuracy = max(
    accuracy_values
)


print("\n================================")
print("BEST K RESULT")
print("================================")

print(
    "Best K:",
    best_k
)

print(
    "Best Accuracy:",
    best_accuracy
)

print(
    "Best Accuracy Percentage:",
    best_accuracy * 100,
    "%"
)


# =========================================================
# 18. ACCURACY VS K VISUALIZATION
# =========================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    k_values,
    accuracy_values,
    marker='o'
)

plt.scatter(
    best_k,
    best_accuracy,
    s=100
)

plt.xlabel(
    "K Value"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "KNN Accuracy vs K Value"
)

plt.xticks(
    k_values
)

plt.grid(
    True
)

plt.show()