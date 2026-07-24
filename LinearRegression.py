from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5]]
Y = [40, 50, 65, 75, 90]

modal = LinearRegression()

modal.fit(X, Y)
hours = float(input("Enter how many hours you studies: "))

predicted_marks = modal.predict([[hours]])

print(f"Based on your hours {hours} you may score around {predicted_marks}")