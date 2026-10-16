from sklearn.linear_model import LogisticRegression

X = [[25, 3000], [40, 5000], [22, 1800], [55, 7000], [30, 4000]]
y = [1, 0, 1, 0, 0]

model = LogisticRegression()
model.fit(X, y)

age = int(input("Age: "))
income = float(input("Monthly income: "))

probability = model.predict_proba([[age, income]])[0, 1]
print("Class probability:", round(probability, 3))
