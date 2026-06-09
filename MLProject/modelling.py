import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_model():
    # Simpan hasil MLflow ke folder lokal repository
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Telco_Customer_Churn_Experiment")

    # Path relatif agar bisa dijalankan di GitHub Actions
# Menggunakan Current Working Directory (CWD) setelah perintah 'cd' dijalankan
    BASE_DIR = os.getcwd()
    data_dir = os.path.join(BASE_DIR, "namadataset_preprocessing")
    print("Data dir:", data_dir)
    print("Folder exists:", os.path.exists(data_dir))
    print("X_train exists:", os.path.exists(os.path.join(data_dir, "X_train.csv")))

    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    
    # Hyperparameter yang akan dicoba (Simulasi Manual Tuning untuk Kriteria Skilled)
    grid_params = [
        {"n_estimators": 50, "max_depth": 5},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 150, "max_depth": 15}
    ]
    
    print("Memulai eksperimen dengan MLflow Manual Logging...")
    
    for i, params in enumerate(grid_params):
        # Memulai run MLflow untuk setiap kombinasi hyperparameter
        with mlflow.start_run(run_name=f"RandomForest_Run_{i+1}"):
            
            # Inisialisasi dan latih model
            model = RandomForestClassifier(
                n_estimators=params["n_estimators"], 
                max_depth=params["max_depth"], 
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Prediksi
            y_pred = model.predict(X_test)
            
            # Hitung Metrik Evaluasi
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # --- MANUAL LOGGING (Syarat Skor Skilled) ---
            # 1. Log Hyperparameter
            mlflow.log_params(params)
            
            # 2. Log Metrik Performa
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            
            # 3. Log Artefak Tambahan: Buat & Simpan Confusion Matrix Plot
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f"Confusion Matrix (Run {i+1})")
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            
            # Simpan plot secara lokal sementara
            plot_path = f"confusion_matrix_run_{i+1}.png"
            plt.savefig(plot_path)
            plt.close()
            
            # Unggah file plot sebagai artefak ke MLflow Tracking UI
            mlflow.log_artifact(plot_path)
            
            # Hapus plot lokal setelah diunggah agar bersih
            if os.path.exists(plot_path):
                os.remove(plot_path)
                
            # 4. Log Model itu sendiri ke dalam Registry Artefak
            mlflow.sklearn.log_model(model, "model")
            
            print(f"Run {i+1} selesai. Parameter: {params} | Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_model()