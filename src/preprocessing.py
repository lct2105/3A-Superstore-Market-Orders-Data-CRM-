# PREPROCESSING MODULE
# Xử lý dữ liệu gốc: đọc file, làm sạch, chuẩn bị tập train-test


import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_excel(path)
    return df

def clean_data(df):
    df = df.dropna()
    return df

def split_data(df, target_col='Profit', test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
