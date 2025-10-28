# MODEL TRAINING MODULE
# Tạo pipelines cho 4 mô hình và train

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def create_models(preprocessor):
    linreg = Pipeline([
        ('prep', preprocessor),
        ('model', LinearRegression())
    ])

    tree = Pipeline([
        ('prep', preprocessor),
        ('model', DecisionTreeRegressor(max_depth=8, random_state=42))
    ])

    rf = Pipeline([
        ('prep', preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=100, max_depth=10,
            random_state=42, n_jobs=-1))
    ])

    xgb_model = Pipeline([
        ('prep', preprocessor),
        ('model', xgb.XGBRegressor(
            n_estimators=100, max_depth=6,
            learning_rate=0.1,
            random_state=42, n_jobs=-1))
    ])

    return {'Linear': linreg, 'DecisionTree': tree, 'RandomForest': rf, 'XGBoost': xgb_model}
