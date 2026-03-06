import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import dagshub

# 1. Inisialisasi DagsHub & MLflow
dagshub.init(repo_owner='riyandimuhamad', repo_name='Eksperimen_SML_Riyandi', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/riyandimuhamad/Eksperimen_SML_Riyandi.mlflow")

# Nama eksperimen
mlflow.set_experiment("Telco_Churn_Tuning")

# 2. Membaca Dataset Bersih
print("Membaca dataset...")
df = pd.read_csv('telco_preprocessing/telco_cleaned.csv')

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Setup Model & Hyperparameter Tuning (Syarat Advanced)
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)

print("Memulai proses training dan tuning...")

# 4. Memulai eksekusi MLflow (MANUAL LOGGING - Syarat Advanced)
with mlflow.start_run():
    # Proses Training
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Prediksi
    y_pred = best_model.predict(X_test)
    
    # Hitung Metrik
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Training Selesai! Akurasi Model Terbaik: {acc:.4f}")
    
    # PROSES MANUAL LOGGING KE DAGSHUB
    
    # A. Log Hyperparameter Terbaik
    mlflow.log_params(grid_search.best_params_)
    
    # B. Log Metrik
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    
    # C. Log Model Utama
    mlflow.sklearn.log_model(best_model, "random_forest_model")
    
    # D. Log Artefak Tambahan 1: Gambar Confusion Matrix (.png)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    
    # E. Log Artefak Tambahan 2: Data Feature Importance (.csv)
    feature_imp = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    feature_imp.to_csv('feature_importance.csv', index=False)
    mlflow.log_artifact('feature_importance.csv')
    
    print("Model, Metrik, dan Artefak berhasil diunggah ke DagsHub!")
