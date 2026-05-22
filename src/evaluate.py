import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").values.ravel()

model = joblib.load("models/model.pkl")
preds = model.predict(X_test)

print("MSE:     ", mean_squared_error(y_test, preds))
print("R² Score:", r2_score(y_test, preds))