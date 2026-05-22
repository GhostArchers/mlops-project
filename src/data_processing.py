import pandas as pd
import yaml
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_regression

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

test_size    = params["prepare"]["test_size"]
random_state = params["prepare"]["random_state"]
percentile   = params["prepare"]["feature_percentile"]

data = pd.read_csv("data/data.csv")

X = data[['Area', 'Sensing Range', 'Transmission Range', 'Number of Sensor nodes']]
y = data['Number of Barriers']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

selector = SelectPercentile(score_func=f_regression, percentile=percentile)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected  = selector.transform(X_test_scaled)

pd.DataFrame(X_train_selected).to_csv("data/X_train.csv", index=False)
pd.DataFrame(X_test_selected).to_csv("data/X_test.csv",  index=False)
pd.DataFrame(y_train).to_csv("data/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("data/y_test.csv",  index=False)

joblib.dump(scaler,   "models/scaler.pkl")
joblib.dump(selector, "models/selector.pkl")

print("Data processing done.")