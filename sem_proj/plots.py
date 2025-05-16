# Generate visualizations like feature importance
import matplotlib.pyplot as plt
import joblib

def plot_feature_importance():
    model = joblib.load("models/random_forest.pkl")
    feature_names = ["danceability", "energy", "valence", "tempo"]
    importances = model.feature_importances_

    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, importances)
    plt.xlabel("Feature Importance")
    plt.title("Random Forest - Feature Importances")
    plt.tight_layout()
    plt.savefig("reports/figures/feature_importance.png")
    plt.show()
