import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report)
from sklearn.feature_selection import SelectKBest, f_classif


# 1. Load the dataset
df = pd.read_csv("KNNAlgorithmDataset.csv")


# 2. Drop unnecessary columns
df = df.drop(['id', 'Unnamed: 32'], axis=1)


# 3. Separate features and target
x = df.drop('diagnosis', axis=1)
y = df['diagnosis']


# 4. Convert target values into numbers
# M = 1 (Malignant)
# B = 0 (Benign)
y = y.map({'M': 1, 'B': 0})


# 5. Split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 6. Select best 5 features
selector = SelectKBest(
    score_func=f_classif,
    k=5
)

X_train_selected = selector.fit_transform(
    x_train,
    y_train
)

X_test_selected = selector.transform(
    x_test
)


# 7. Get selected feature names
selected_features = x.columns[
    selector.get_support()
]

print("Selected Features:")
print(selected_features)


# 8. Scale selected features
scaler = StandardScaler()

X_train_selected = scaler.fit_transform(
    X_train_selected
)

X_test_selected = scaler.transform(
    X_test_selected
)


# 9. Create KNN model
knn = KNeighborsClassifier(
    n_neighbors=5
)


# 10. Train KNN model
knn.fit(
    X_train_selected,
    y_train
)


# 11. Make predictions
y_pred = knn.predict(
    X_test_selected
)


# 12. Calculate accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:", accuracy)


# 13. Confusion Matrix
print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# 14. Classification Report
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# 15. Take user input
print("\nEnter values for prediction:")

user_input = []

for feature in selected_features:

    value = float(
        input(f"Enter {feature}: ")
    )

    user_input.append(value)


# 16. Scale user input
user_input_scaled = scaler.transform(
    [user_input]
)


# 17. Predict
prediction = knn.predict(
    user_input_scaled
)


# 18. Display result
if prediction[0] == 1:

    print("\nPrediction: Malignant")

else:

    print("\nPrediction: Benign")

df[selected_features].hist(
    figsize=(12, 8)
)

plt.suptitle("Distribution of Selected Features")

plt.show()