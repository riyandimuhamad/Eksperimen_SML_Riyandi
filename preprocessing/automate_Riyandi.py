import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def load_data(file_path):
    print("Membaca data mentah...")
    return pd.read_csv(file_path)

def preprocess_data(df):
    print("Memulai proses pembersihan data...")
    # 1. Hapus kolom yang tidak perlu
    df = df.drop('customerID', axis=1)
    
    # 2. Perbaiki tipe data
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # 3. Isi nilai kosong dengan median
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # 4. Encoding
    binary_cols = [col for col in df.columns if df[col].nunique() == 2 and col != 'Churn']
    multi_cols = [col for col in df.columns if df[col].nunique() > 2 and col not in ['tenure', 'MonthlyCharges', 'TotalCharges']]
    
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])
    
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
    df = df.astype(float)
    
    print("Pembersihan data selesai.")
    return df

def save_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data bersih berhasil disimpan di: {output_path}")

if __name__ == "__main__":
    # Menentukan path yang dinamis agar bisa jalan di GitHub Actions
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, 'data_raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    output_file = os.path.join(base_dir, 'preprocessing', 'telco_preprocessing', 'telco_cleaned.csv')
    
    raw_data = load_data(input_file)
    cleaned_data = preprocess_data(raw_data)
    save_data(cleaned_data, output_file)