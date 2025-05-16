import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sem_proj.dataset import load_data
import os

def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_data()

    
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    print("Linear Regression:")
    print("R²:", r2_score(y_test, lr_preds))
    print("MAE:", mean_absolute_error(y_test, lr_preds))

    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    print("\nRandom Forest:")
    print("R²:", r2_score(y_test, rf_preds))
    print("MAE:", mean_absolute_error(y_test, rf_preds))

    
    os.makedirs("models", exist_ok=True)
    joblib.dump(lr_model, "models/linear_model.pkl")
    joblib.dump(rf_model, "models/random_forest.pkl")

if __name__ == "__main__":
    train_and_evaluate()
#Train LinearRegression and RandomForest models
