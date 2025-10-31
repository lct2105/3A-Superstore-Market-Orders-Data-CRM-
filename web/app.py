# SUPERSTORE WEB
import os
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# Page config & styles
st.set_page_config(page_title="Superstore Dashboard", page_icon="🏪", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f7f9fc; }
    .stButton>button {
        background-color: #2e86de; color: white; font-weight: 600;
        border-radius: 8px; height: 3em; padding: 0 1.2em;
    }
    .stButton>button:hover { background-color: #1b4f72; }
    .card { background:#e8f5e9; padding:18px; border-radius:12px; border:1px solid #4caf50; }
    .card h3 { color:#2e7d32; margin:0 0 6px 0; text-align:center; }
    .card h1 { color:#1b5e20; margin:0; text-align:center; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="background-color:#2e86de;padding:18px;border-radius:10px;text-align:center">
        <h2 style="color:white;margin:0;">Superstore Profit Prediction Dashboard</h2>
        <p style="color:#eaf2f8;margin:6px 0 0;">
            Upload/EDA/Predict — tối ưu hiệu năng & tránh giật/đơ
        </p>
    </div>
""", unsafe_allow_html=True)

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "Superstore.xls")
MODEL_PATH = os.path.join(BASE_DIR, "..", "model_XGBoost.pkl")  
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")


# Cache helpers
@st.cache_data(show_spinner=False)
def list_available_models(models_dir):
    if not os.path.exists(models_dir):
        return {}
    files = [f for f in os.listdir(models_dir) if f.startswith("model_") and f.endswith(".pkl")]
    mapping = {}
    for f in files:
        name = f.replace("model_", "").replace(".pkl", "")
        mapping[name] = os.path.join(models_dir, f)
    return mapping

@st.cache_resource(show_spinner=False)
def load_model_by_path(path):
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def load_df(file_or_path):
    if isinstance(file_or_path, str):
        path = file_or_path
        if path.endswith(".csv"):
            return pd.read_csv(path)
        return pd.read_excel(path)
    else:
        # Uploaded file-like
        name = getattr(file_or_path, "name", "")
        if name.endswith(".csv"):
            return pd.read_csv(file_or_path)
        return pd.read_excel(file_or_path)

@st.cache_resource(show_spinner=False)
def load_model(path):
    return joblib.load(path)

# Keep DF in session
def set_df(df):
    st.session_state["df"] = df

def get_df():
    return st.session_state.get("df", None)

# Sidebar controls
with st.sidebar:
    st.markdown("### Cài đặt hiển thị")
    sample_n = st.slider("Số dòng mẫu cho biểu đồ (giảm lag)", 500, 5000, 3000, 500)
    show_raw = st.checkbox("Hiển thị preview dữ liệu (50 dòng)", value=True)

# Tabs
tab_data, tab_eda, tab_pred = st.tabs(["DATA", "TỔNG QUAN DỮ LIỆU", "DỰ ĐOÁN"])

# TAB 1 — DATA
with tab_data:
    st.markdown("#### Chọn dữ liệu")
    up = st.file_uploader("Tải lên (.xls/.xlsx/.csv)", type=["xls", "xlsx", "csv"])
    colA, colB = st.columns([1,1])
    with colA:
        if st.button("Dùng data mẫu (Suoperstoe.xls)", use_container_width=True):
            df = load_df(DATA_PATH)
            set_df(df)
            st.success("Đã nạp file mẫu")
    with colB:
        if up is not None:
            df = load_df(up)
            set_df(df)
            st.success(f"Đã nạp file tải lên: {getattr(up,'name','uploaded file')}")

    df = get_df()
    if df is None:
        # Tự nạp file mẫu lần đầu
        df = load_df(DATA_PATH)
        set_df(df)

    st.markdown("##### Thông tin dữ liệu")
    st.write(f"- Kích thước: **{df.shape[0]} dòng × {df.shape[1]} cột**")
    if show_raw:
        st.write("**Xem trước 50 dòng:**")
        st.dataframe(df.sample(min(50, len(df)), random_state=42))

# TAB 2 — EDA
with tab_eda:
    df = get_df()
    if df is None:
        st.warning("Chưa có dữ liệu. Vui lòng vào tab **Data** để nạp dữ liệu.")
    else:
        st.markdown("#### Tổng quan & biểu đồ (dùng mẫu để tránh giật)")
        # Sample gọn
        df_sample = df.sample(n=min(sample_n, len(df)), random_state=42)

        # Ensure cột cần có
        needed_cols = ["Category", "Sales", "Region", "Profit", "Quantity", "Sub-Category"]
        miss = [c for c in needed_cols if c not in df_sample.columns]
        if miss:
            st.error(f"Thiếu cột: {miss}. Kiểm tra lại file dữ liệu.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Doanh thu theo danh mục (Category)")
                g1 = df_sample.groupby("Category", as_index=False)["Sales"].sum()
                fig1 = px.bar(g1, x="Category", y="Sales", color="Category",
                              title="Tổng doanh thu theo Category", text_auto=True)
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                st.subheader("Lợi nhuận theo vùng (Region)")
                g2 = df_sample.groupby("Region", as_index=False)["Profit"].sum()
                fig2 = px.bar(g2, x="Region", y="Profit", color="Region",
                              title="Tổng lợi nhuận theo Region", text_auto=True)
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Sales ↔ Profit (sample)")
            fig3 = px.scatter(
                df_sample, x="Sales", y="Profit", color="Category",
                size="Quantity", opacity=0.75,
                hover_data=["Sub-Category"], title="Mối quan hệ Sales vs Profit"
            )
            fig3.update_layout(transition_duration=0)
            st.plotly_chart(fig3, use_container_width=True)

# TAB 3 — PREDICT
with tab_pred:
    df = get_df()
    if df is None:
        st.warning("Chưa có dữ liệu. Vui lòng vào tab **Data** để nạp dữ liệu.")
        st.stop()

    st.markdown("#### Chọn mô hình & nhập thông tin để dự đoán Profit")

    # 🔎 Tìm các mô hình đã save
    model_map = list_available_models(MODELS_DIR)  # dict: {"Linear": ".../model_Linear.pkl", ...}

    if not model_map:
        # fallback về file cũ ở gốc (best model)
        if not os.path.exists(MODEL_PATH):
            st.error("Không tìm thấy mô hình nào. Hãy chạy lại `src/main.py` để huấn luyện & lưu model.")
            st.stop()
        st.info("Không thấy thư mục models/. Đang dùng mô hình mặc định (best) đã lưu ở gốc.")
        selected_name = "XGBoost"  # tên hiển thị
        model = load_model(MODEL_PATH)
    else:
        # Ưu tiên XGBoost nếu có, không thì lấy cái đầu
        default_name = "XGBoost" if "XGBoost" in model_map else sorted(model_map.keys())[0]
        selected_name = st.selectbox("Chọn mô hình để dự đoán", sorted(model_map.keys()), index=sorted(model_map.keys()).index(default_name))
        model = load_model_by_path(model_map[selected_name])


    # Lựa chọn từ data hiện có
    def safe_unique(col, fallback=None):
        return sorted(df[col].dropna().unique().tolist()) if col in df.columns else (fallback or [])

    c1, c2, c3 = st.columns(3)
    with c1:
        sales = st.number_input("Sales", min_value=0.0, value=200.0, step=10.0)
    with c2:
        quantity = st.number_input("Quantity", min_value=1, value=2, step=1)
    with c3:
        discount = st.slider("Discount", 0.0, 0.9, 0.1)

    c4, c5, c6 = st.columns(3)
    with c4:
        category = st.selectbox("Category", safe_unique("Category", ["Furniture","Office Supplies","Technology"]))
    with c5:
        sub_category = st.selectbox("Sub-Category", safe_unique("Sub-Category", ["Phones","Chairs","Storage"]))
    with c6:
        region = st.selectbox("Region", safe_unique("Region", ["West","East","Central","South"]))

    input_df = pd.DataFrame({
        "Sales": [sales],
        "Quantity": [quantity],
        "Discount": [discount],
        "Category": [category],
        "Sub-Category": [sub_category],
        "Region": [region],
    })

    # Bổ sung cột còn thiếu dựa trên preprocessor của pipeline
    if hasattr(model, "named_steps") and "prep" in model.named_steps:
        preprocessor = model.named_steps["prep"]
        num_cols, cat_cols = [], []
        # giả định bạn đã đặt tên 'num' & 'cat' trong ColumnTransformer
        for name, transformer, cols in preprocessor.transformers_:
            if name == "num":
                num_cols += cols
            elif name == "cat":
                cat_cols += cols
        expected_cols = list(dict.fromkeys(num_cols + cat_cols))  # preserve order, unique
        # add missing with correct dtype defaults
        for col in expected_cols:
            if col not in input_df.columns:
                input_df[col] = 0 if col in num_cols else "missing"
        # re-order to match training columns (không bắt buộc, nhưng tốt)
        input_df = input_df.reindex(columns=expected_cols, fill_value=0)
    else:
        st.info("Model không có bước 'prep' trong pipeline — dùng trực tiếp features hiện tại.")

    if st.button("Dự đoán lợi nhuận", use_container_width=True):
        with st.spinner("Đang dự đoán..."):
            pred = float(model.predict(input_df)[0])

        st.markdown(f"""
            <div class="card">
                <h3>Lợi nhuận dự đoán — <span style="font-weight:600;">{selected_name}</span></h3>
                <h1>{pred:.2f} USD</h1>
            </div>
        """, unsafe_allow_html=True)

        # Vẽ vị trí dự đoán trên scatter (dùng sample để nhẹ)
        df_sample = df.sample(n=min(2000, len(df)), random_state=42) if len(df) > 2000 else df.copy()
        figp = px.scatter(
            df_sample, x="Sales", y="Profit",
            color="Category", opacity=0.5, title="Vị trí điểm dự đoán trên Sales-Profit (sample)"
        )
        figp.add_scatter(x=[sales], y=[pred],
                         mode="markers+text", text=["🔹 Dự đoán"], textposition="top center",
                         marker=dict(size=16, color="red", symbol="star"))
        st.plotly_chart(figp, use_container_width=True)
