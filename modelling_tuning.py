import pandas as pd
import mlflow
import mlflow.sklearn
import shutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.environ['MLFLOW_TRACKING_USERNAME'] = 'Zahran-Fardiaz' 
os.environ['MLFLOW_TRACKING_PASSWORD'] = '7bb01bb9378f73113164f2f2912e402ee595b970'

DAGSHUB_URI = 'https://dagshub.com/Zahran-Fardiaz/Eksperimen_SML_Zahran-Fardiaz.mlflow'
mlflow.set_tracking_uri(DAGSHUB_URI)

def load_processed_data():
    X_train = pd.read_csv("membangun_model/datasets_preprocessing/X_train_clean.csv")
    X_test = pd.read_csv("membangun_model/datasets_preprocessing/X_test_clean.csv")
    y_train = pd.read_csv("membangun_model/datasets_preprocessing/y_train_clean.csv").values.ravel()
    y_test = pd.read_csv("membangun_model/datasets_preprocessing/y_test_clean.csv").values.ravel()
    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = load_processed_data()

    mlflow.set_experiment("Credit_Card_Approval_Tuning")

    with mlflow.start_run():
        print("Melakukan Hyperparameter Tuning & Logging ke DagsHub")
        
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None],
            'random_state': [42]
        }
        
        rf = RandomForestClassifier()
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        mlflow.log_params(grid_search.best_params_)

        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(best_model, "model_random_forest")

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix - Credit Card Approval")
        plt.ylabel('Aktual')
        plt.xlabel('Prediksi')
        cm_filename = "confusion_matrix.png"
        plt.savefig(cm_filename)
        mlflow.log_artifact(cm_filename)

        if os.path.exists("saved_model"):
            shutil.rmtree("saved_model")
        mlflow.sklearn.save_model(best_model, "saved_model")
        
        print(f"Pelatihan online selesai. Akurasi Model: {acc:.4f}")

if __name__ == "__main__":
    main()