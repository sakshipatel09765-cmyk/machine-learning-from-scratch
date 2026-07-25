import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# -----------------------------
# 1. Dataset Load
# -----------------------------
df = pd.read_csv("StudentsPerformance.csv")

# -----------------------------
# 2. Features and Target
# -----------------------------
X = df[["reading score", "writing score"]]
y = df["math score"]

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# 4. Train Linear Regression
# -----------------------------
model = LinearRegression()

model.fit(X_train, y_train)

print("Model Training Complete!")

# -----------------------------
# 5. User se Input Lena
# -----------------------------
reading = float(input("Enter Reading Score: "))
writing = float(input("Enter Writing Score: "))

# -----------------------------
# 6. Prediction
# -----------------------------
new_data = pd.DataFrame({
    "reading score": [reading],
    "writing score": [writing]
})

prediction = model.predict(new_data)

# -----------------------------
# 7. Result
# -----------------------------
print("\nPredicted Math Score:", round(prediction[0], 2))