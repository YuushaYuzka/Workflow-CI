import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def train_model():
    # Mengunci direktori dasar ke tempat skrip berada secara mutlak
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Atur Tracking URI lokal di dalam folder instansiasi skrip
    mlruns_dir = os.path.join(BASE_DIR, "mlruns")
    mlflow.set_tracking_uri(f"file:{mlruns_dir}")
    mlflow.set_experiment("Telco_Customer_Churn_Experiment")

    # 2. Ambil jalur data secara dinamis
    # Jika dijalankan via 'mlflow run', kita paksa membaca direktori asal eksekusi awal
    data_dir = os.path.join(BASE_DIR, "namadataset_preprocessing")
    
    print(f"[DEBUG] Menjalankan kode dari lokasi: {BASE_DIR}")
    print(f"[DEBUG] Mencari direktori data di: {data_dir}")

    if not os.path.exists(data_dir):
        print(f"[WARNING] Jalur {data_dir} tidak ada. Mencoba fallback ke direktori kerja saat ini...")
        data_dir = os.path.join(os.getcwd(), "namadataset_preprocessing")

    # Memuat file dataset
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    
    grid_params = [
        {"n_estimators": 50, "max_depth": 5},
        {"n_estimators": 100, "max_depth": 10}
    ]
    
    print("Memulai eksperimen dengan MLflow Manual Logging...")
    for i, params in enumerate(grid_params):
        with mlflow.start_run(run_name=f"RandomForest_Run_{i+1}"):
            model = RandomForestClassifier(
                n_estimators=params["n_estimators"], 
                max_depth=params["max_depth"], 
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Manual Logging untuk kriteria Skilled
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            
            # Artefak Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.close()
            
            mlflow.sklearn.log_model(model, "model")
            print(f"Run {i+1} sukses. Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_model()