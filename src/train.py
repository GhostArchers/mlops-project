import pandas as pd
import yaml
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

random_state = params["model"]["random_state"]

X_train = pd.read_csv("data/X_train.csv")
X_test  = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test  = pd.read_csv("data/y_test.csv").values.ravel()

mlflow.set_experiment("Barrier_Prediction")

with mlflow.start_run():

    param_dist = {
        'n_estimators':  [50, 100, 200, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.5],
        'max_depth':     [3, 4, 5, 6],
        'subsample':     [0.6, 0.8, 1.0]
    }

    search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=random_state),
        param_dist, n_iter=20, cv=5,
        random_state=random_state, n_jobs=-1
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    preds      = best_model.predict(X_test)
    mse        = mean_squared_error(y_test, preds)
    r2         = r2_score(y_test, preds)

    mlflow.log_params(search.best_params_)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2",  r2)

    joblib.dump(best_model, "models/model.pkl")
    mlflow.sklearn.log_model(best_model, "model")

print("Training done.")