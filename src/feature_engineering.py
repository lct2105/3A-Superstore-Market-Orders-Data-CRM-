# FEATURE ENGINEERING MODULE
# Tạo biến mới, mã hóa, chuẩn hóa

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def build_preprocessor():
    num_cols = ['Sales','Quantity','Discount']
    cat_cols = ['Ship Mode','Segment','Region','Category','Sub-Category']

    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, num_cols),
            ("cat", categorical_tf, cat_cols)
        ],
        remainder="drop"
    )
    return preprocessor
