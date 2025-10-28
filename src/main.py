from preprocessing import load_data, clean_data, split_data
from feature_engineering import build_preprocessor
from model_training import create_models
from evaluation import evaluate_reg, compare_models

import os
import joblib

# LOAD & CLEAN DATA

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Superstore.xls')

print("Loading data...")
df = load_data(data_path)  
df = clean_data(df)
print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

# SPLIT DATA

X_train, X_test, y_train, y_test = split_data(df)
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# BUILD PREPROCESSOR

print("Building preprocessor...")
preprocessor = build_preprocessor(X_train)
print("Preprocessor ready!")

# TRAIN MODELS

print("\nTraining all models...")
models = create_models(preprocessor)
results = []

for name, model in models.items():
    print(f"\nTraining model: {name}")
    model.fit(X_train, y_train)
    res = evaluate_reg(name, model, X_test, y_test)
    results.append(res)

# COMPARE RESULTS

comparison_df = compare_models(results)

print("\nFINAL COMPARISON:")
print(comparison_df)


MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

for name, mdl in models.items():
    out_path = os.path.join(MODELS_DIR, f"model_{name}.pkl")
    joblib.dump(mdl, out_path)
    print(f"Saved: {out_path}")

# (tùy chọn) vẫn lưu best model ở gốc để tương thích cũ
best_model_name = comparison_df.loc[comparison_df['R2'].idxmax(), 'Model']
best_model = models[best_model_name]
joblib.dump(best_model, os.path.join(os.path.dirname(__file__), '..', f"model_{best_model_name}.pkl"))
print(f"Best model saved as model_{best_model_name}.pkl")