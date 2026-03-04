import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow

# Menggunakan autolog sesuai syarat Kriteria 2 Basic
mlflow.autolog()

# Membaca Data
df = pd.read_csv('telco_preprocessing/telco_cleaned.csv')
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training sederhana tanpa tuning
with mlflow.start_run(run_name="Basic_Autolog_Run"):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)