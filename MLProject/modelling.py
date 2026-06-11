import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

# 1. Aktifkan Autolog agar otomatis terintegrasi dengan mlflow run
mlflow.autolog()

def train_model():
    print("Memulai pembacaan dataset untuk pipeline CI...")
    
    # Menunjuk ke folder internal MLProject di runner GitHub Actions
    data_dir = "namadataset_preprocessing"
    
    X_train_path = os.path.join(data_dir, "X_train.csv")
    X_test_path = os.path.join(data_dir, "X_test.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")
    y_test_path = os.path.join(data_dir, "y_test.csv")
    
    # Memuat data
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    y_test = pd.read_csv(y_test_path).squeeze()
    
    print("Data berhasil dimuat. Melatih model baseline...")
    
    # PENTING: Jangan gunakan 'with mlflow.start_run()' lagi di sini!
    # Karena mlflow run di GitHub Actions sudah otomatis membuat dan mengelola jalannya RUN.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluasi sederhana untuk memicu metrik autolog
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Retraining Selesai! Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

if __name__ == "__main__":
    train_model()