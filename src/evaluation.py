# EVALUATION MODULE
# Đánh giá và so sánh mô hình

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

def evaluate_reg(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{name}: R2={r2:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}")
    return {'Model': name, 'R2': r2, 'MAE': mae, 'RMSE': rmse}

def compare_models(results_list):
    df = pd.DataFrame(results_list)
    best = df.loc[df['R2'].idxmax()]
    print("\nMô hình ngon nhất:", best['Model'])
    return df
