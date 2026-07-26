import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder

# Dataset
data = {
    "Weather": [
        "Sunny", "Sunny", "Overcast", "Rainy", "Rainy",
        "Overcast", "Sunny", "Rainy", "Sunny", "Overcast"
    ],
    "PlayTennis": [
        "No", "No", "Yes", "Yes", "Yes",
        "Yes", "Yes", "No", "No", "Yes"
    ]
}

df = pd.DataFrame(data)

# Encode Weather
weather_encoder = LabelEncoder()
X = weather_encoder.fit_transform(df["Weather"]).reshape(-1, 1)

# Encode Target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df["PlayTennis"])

# Create and train Naive Bayes model
model = CategoricalNB()
model.fit(X, y)

# Take user input
user_input = input("Enter weather (Sunny/Overcast/Rainy): ")

# Validate input
if user_input not in weather_encoder.classes_:
    print("Invalid weather! Please enter Sunny, Overcast, or Rainy.")
else:
    # Encode user input
    new_data = weather_encoder.transform([user_input]).reshape(-1, 1)

    # Make prediction
    prediction = model.predict(new_data)

    # Convert prediction back to Yes/No
    result = target_encoder.inverse_transform(prediction)

    print("Weather:", user_input)
    print("Will Play Tennis:", result[0])