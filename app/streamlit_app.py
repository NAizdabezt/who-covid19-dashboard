import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import xgboost as xgb
from typing import Tuple

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="WHO COVID-19 Dashboard (Optimized)", page_icon="🌍", layout="wide")
st.title("🌍 WHO COVID-19 Global COVID-19 Dashboard (Optimized)")
st.markdown("Dữ liệu và dashboard tối ưu: cache xử lý, load model sẵn, tắt render nặng khi không cần.")

# ===============================
# HELPERS: path resolution
# ===============================
# Common possible paths (colab, local repo, uploaded)
POSSIBLE_DF_PATHS = [
    "data/df_clean.csv.gz",
    "data/df_clean.csv",
    "/mnt/data/df_clean.csv.gz",
    "/mnt/data/df_clean.csv",
    "/content/drive/MyDrive/dataWHO/df_clean.csv",
    "/content/drive/MyDrive/dataWHO/df_clean.csv.gz"
]
POSSIBLE_LATEST_PATHS = [
    "data/latest.csv.gz",
    "data/latest.csv",
    "/mnt/data/latest (1).csv.gz",      # uploaded file path from this session
    "/mnt/data/latest.csv.gz",
    "/mnt/data/latest.csv",
    "/content/drive/MyDrive/dataWHO/latest.csv",
    "/content/drive/MyDrive/dataWHO/latest.csv.gz"
]

POSSIBLE_MODEL_PATHS = {
    "lr": "app/models/model_lr.pkl",
    "rf": "app/models/model_rf.pkl",
    "xgb_json": "app/models/model_xgb.json"
}

def find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

# ===============================
# 1) Load data (cached)
# ===============================
@st.cache_data(show_spinner=False)
def load_data_cached() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Try df_clean
    df_path = find_first_existing(POSSIBLE_DF_PATHS)
    latest_path = find_first_existing(POSSIBLE_LATEST_PATHS)

    # Fallback: try to read from repo paths (if not present, raise clear error)
    if df_path is None and latest_path is None:
        raise FileNotFoundError(
            "Không tìm thấy file df_clean/latest. Thử upload vào /mnt/data hoặc đặt vào thư mục data/ trong repo."
        )

    # load df (full timeseries)
    if df_path:
        df = pd.read_csv(df_path, parse_dates=["Date_reported"], low_memory=False)
    else:
        # try to construct df from latest? safer to error
        df = pd.DataFrame()
    # load latest (per-country snapshot)
    if latest_path:
        latest = pd.read_csv(latest_path, low_memory=False)
    else:
        latest = pd.DataFrame()

    # Ensure types and column names standardized
    if "Date_reported" in df.columns:
        df["Date_reported"] = pd.to_datetime(df["Date_reported"], errors="coerce")
    # normalize column names (strip)
    df.columns = df.columns.str.strip()
    latest.columns = latest.columns.str.strip()

    return df, latest

# Load data and handle errors
try:
    df, latest = load_data_cached()
except Exception as e:
    st.error(f"Lỗi khi load dữ liệu: {e}")
    st.stop()

# ===============================
# 2) Load models (cached resource)
# ===============================
@st.cache_resource
def load_models():
    models = {}
    # Linear Regression
    p_lr = POSSIBLE_MODEL_PATHS["lr"]
    if os.path.exists(p_lr):
        try:
            models["Linear Regression"] = joblib.load(p_lr)
        except Exception as e:
            models["Linear Regression"] = None
    else:
        models["Linear Regression"] = None
    # Random Forest
    p_rf = POSSIBLE_MODEL_PATHS["rf"]
    if os.path.exists(p_rf):
        try:
            models["Random Forest"] = joblib.load(p_rf)
        except Exception as e:
            models["Random Forest"] = None
    else:
        models["Random Forest"] = None
    # XGBoost JSON
    p_xgb = POSSIBLE_MODEL_PATHS["xgb_json"]
    if os.path.exists(p_xgb):
        try:
            m = xgb.XGBRegressor()
            m.load_model(p_xgb)
            models["XGBoost"] = m
        except Exception as e:
            models["XGBoost"] = None
    else:
        models["XGBoost"] = None

    return models

MODELS = load_models()

# ===============================
# 3) Sidebar: filters & perf toggles
# ===============================
st.sidebar.header("🔎 Bộ lọc & Tùy chọn (Optimized)")
# country selector
countries = sorted(df["Country"].dropna().unique()) if "Country" in df.columns else []
selected_country = st.sidebar.selectbox("Chọn quốc gia", ["Toàn cầu"] + countries)

# date range selector with safe defaults
if "Date_reported" in df.columns and not df["Date_reported"].isna().all():
    min_ts = df["Date_reported"].min()
    max_ts = df["Date_reported"].max()
else:
    min_ts = pd.Timestamp("2020-01-01")
    max_ts = pd.Timestamp.today()

st.sidebar.caption(f"Khoảng dữ liệu: {min_ts.date()} → {max_ts.date()}")

date_input = st.sidebar.date_input("Chọn khoảng thời gian (nhấn 2 lần để chọn range)", value=(min_ts.date(), max_ts.date()))
# normalize date_input to start_ts, end_ts
if isinstance(date_input, (list, tuple)) and len(date_input) == 2:
    start_ts = pd.to_datetime(date_input[0])
    end_ts = pd.to_datetime(date_input[1])
else:
    start_ts = min_ts
    end_ts = max_ts
# clamp
start_ts = max(start_ts, min_ts)
end_ts = min(end_ts, max_ts)
if start_ts > end_ts:
    start_ts, end_ts = end_ts, start_ts

# Performance toggles
show_map_2d = st.sidebar.checkbox("Hiển thị bản đồ 2D", value=True)
show_map_3d = st.sidebar.checkbox("Hiển thị bản đồ 3D (nặng)", value=False)
show_top_table = st.sidebar.checkbox("Hiển thị Top quốc gia", value=True)
show_ml = st.sidebar.checkbox("Hiển thị tab ML (load models)", value=False)

# ===============================
# 4) Preprocess & cached aggregations
# ===============================
@st.cache_data
def filter_df_by_date(df_in: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if "Date_reported" not in df_in.columns:
        return df_in.copy()
    mask = (df_in["Date_reported"] >= start) & (df_in["Date_reported"] <= end)
    return df_in.loc[mask].copy()

df_filtered = filter_df_by_date(df, start_ts, end_ts)

@st.cache_data
def compute_latest_aggregates(df_filtered: pd.DataFrame, latest_original: pd.DataFrame) -> pd.DataFrame:
    # groupby only once; keep necessary columns
    if df_filtered.empty:
        return pd.DataFrame()
    grp = df_filtered.groupby("Country", as_index=False).agg({
        "New_cases": "sum",
        "New_deaths": "sum"
    }).rename(columns={"New_cases": "Cumulative_cases", "New_deaths": "Cumulative_deaths"})
    # merge population & iso3 if present in latest_original
    if {"Country","Country_code3","Population"}.issubset(latest_original.columns):
        meta = latest_original[["Country","Country_code3","Population"]].drop_duplicates(subset=["Country"])
        merged = grp.merge(meta, on="Country", how="left")
    else:
        merged = grp
    # safe computations
    if "Population" in merged.columns:
        merged["Cases_per_million"] = merged.apply(
            lambda r: r["Cumulative_cases"] / (r["Population"]/1_000_000) if pd.notnull(r.get("Population")) and r.get("Population")>0 else np.nan,
            axis=1
        )
    else:
        merged["Cases_per_million"] = np.nan
    merged["Fatality_rate"] = merged.apply(
        lambda r: (r["Cumulative_deaths"]/r["Cumulative_cases"]*100) if r["Cumulative_cases"]>0 else np.nan,
        axis=1
    )
    return merged

latest_filtered = compute_latest_aggregates(df_filtered, latest)

# ===============================
# 5) KPI (fast)
# ===============================
col1, col2, col3, col4 = st.columns(4)
total_cases = int(latest_filtered["Cumulative_cases"].sum()) if not latest_filtered.empty else 0
total_deaths = int(latest_filtered["Cumulative_deaths"].sum()) if not latest_filtered.empty else 0
fatality_rate_overall = (total_deaths/total_cases*100) if total_cases>0 else 0
affected_countries = int(latest_filtered["Country"].nunique()) if not latest_filtered.empty else 0

col1.metric("🦠 Tổng ca nhiễm (giai đoạn)", f"{total_cases:,}")
col2.metric("⚰️ Tổng ca tử vong (giai đoạn)", f"{total_deaths:,}")
col3.metric("📊 Tỷ lệ tử vong (%)", f"{fatality_rate_overall:.2f}")
col4.metric("🌎 Quốc gia (ghi nhận trong giai đoạn)", f"{affected_countries}")

# ===============================
# 6) Main content (Tabs)
# ===============================
tab1, tab2, tab3 = st.tabs(["📈 Xu hướng", "🗺️ Bản đồ", "📊 Top & ML"])

# --- TAB 1: Trends ---
with tab1:
    st.subheader("📈 Xu hướng ca nhiễm theo thời gian")
    if selected_country != "Toàn cầu":
        df_country = df_filtered[df_filtered["Country"]==selected_country]
        fig_line = px.line(df_country, x="Date_reported", y="New_cases", title=f"Số ca mới - {selected_country}", labels={"New_cases":"Ca mới","Date_reported":"Ngày"})
    else:
        agg = df_filtered.groupby("Date_reported", as_index=False)[["New_cases","New_deaths"]].sum()
        fig_line = px.line(agg, x="Date_reported", y="New_cases", title="Số ca mới toàn cầu (giai đoạn)", labels={"New_cases":"Ca mới","Date_reported":"Ngày"})
    st.plotly_chart(fig_line, use_container_width=True)

# --- TAB 2: Maps ---
with tab2:
    st.subheader("🗺️ Bản đồ (tùy chọn hiển thị)")
    if latest_filtered.empty:
        st.info("Không có dữ liệu để vẽ bản đồ trong khoảng ngày hiện tại.")
    else:
        if show_map_2d:
            st.markdown("#### Bản đồ 2D — Tổng ca (theo khoảng ngày)")
            fig2 = px.choropleth(
                latest_filtered,
                locations="Country_code3" if "Country_code3" in latest_filtered.columns else "Country",
                color="Cumulative_cases",
                hover_name="Country",
                color_continuous_scale="Reds",
                projection="natural earth"
            )
            fig2.update_layout(margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.caption("Bản đồ 2D bị tắt (bật lại ở sidebar).")

        if show_map_3d:
            st.markdown("#### Bản đồ 3D (Interactive globe) — *có thể nặng*")
            fig3 = go.Figure(go.Choropleth(
                locations=latest_filtered["Country_code3"] if "Country_code3" in latest_filtered.columns else latest_filtered["Country"],
                z=latest_filtered["Cases_per_million"] if "Cases_per_million" in latest_filtered.columns else latest_filtered["Cumulative_cases"],
                text=latest_filtered["Country"] if "Country" in latest_filtered.columns else None,
                colorscale="Reds",
                marker_line_color="black",
                marker_line_width=0.3,
            ))
            fig3.update_geos(projection_type="orthographic", showocean=True, showcountries=True)
            fig3.update_layout(height=600, margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.caption("Bản đồ 3D tắt (bật nếu muốn).")

# --- TAB 3: Top & ML (split)
with tab3:
    left, right = st.columns([2,1])
    with left:
        st.subheader("🏆 Top quốc gia (theo khoảng ngày)")
        if latest_filtered.empty:
            st.info("Không có dữ liệu Top.")
        else:
            metric_opt = st.selectbox("Chọn metric", ["Cumulative_cases","Cumulative_deaths","Cases_per_million","Fatality_rate"])
            df_plot = latest_filtered.copy()
            df_plot[metric_opt] = pd.to_numeric(df_plot[metric_opt], errors="coerce").fillna(0)
            top10 = df_plot.nlargest(10, metric_opt)
            fig_top = px.bar(
                top10.sort_values(metric_opt, ascending=True),
                x=metric_opt, y="Country", orientation="h",
                text=metric_opt, color=metric_opt, color_continuous_scale="Reds",
                labels={metric_opt: metric_opt, "Country":"Quốc gia"}, height=520
            )
            fig_top.update_traces(texttemplate="%{text:,.0f}" if metric_opt!="Fatality_rate" else "%{text:.2f}%", textposition="outside")
            fig_top.update_layout(margin=dict(l=70,r=80,t=60,b=20), plot_bgcolor="white")
            st.plotly_chart(fig_top, use_container_width=True)

    with right:
        st.subheader("🔮 Machine Learning (Load model sẵn)")
        if not show_ml:
            st.info("Bật ML ở sidebar để thao tác với mô hình đã train.")
        else:
            # show available models
            avail = {k:v for k,v in MODELS.items() if v is not None}
            if not avail:
                st.warning("Không có model đã train trong thư mục app/models. Hãy upload model_lr.pkl, model_rf.pkl, model_xgb.json")
            else:
                model_name = st.selectbox("Chọn model", list(avail.keys()))
                model = avail[model_name]
                country_ml = st.selectbox("Chọn quốc gia để forecast", countries, index=countries.index("Viet Nam") if "Viet Nam" in countries else 0)
                horizon_ml = st.slider("Số ngày dự báo", 7, 30, value=14)

                # cached features per country
                @st.cache_data
                def make_features_cached(country_name):
                    d = df[df["Country"]==country_name].sort_values("Date_reported").reset_index(drop=True)
                    if d.empty:
                        return pd.DataFrame()
                    d["lag_1"] = d["New_cases"].shift(1)
                    d["lag_7"] = d["New_cases"].shift(7)
                    d["lag_14"] = d["New_cases"].shift(14)
                    d["ma7"] = d["New_cases"].rolling(7).mean().shift(1)
                    d["ma14"] = d["New_cases"].rolling(14).mean().shift(1)
                    d["weekday"] = d["Date_reported"].dt.weekday
                    d = d.dropna().reset_index(drop=True)
                    return d

                df_feat = make_features_cached(country_ml)
                if df_feat.empty or len(df_feat) < 20:
                    st.warning("Dữ liệu thời gian của quốc gia này quá ít để dự báo (cần ≥ 20).")
                else:
                    if st.button("Chạy dự báo (load model)"):
                        # iterative forecast using cached model
                        def iterative_forecast_local(model_local, df_local, horizon_local):
                            last = df_local.iloc[-1]
                            lag1, lag7, lag14 = last["lag_1"], last["lag_7"], last["lag_14"]
                            recent = list(df_local["New_cases"].iloc[-14:].values)
                            preds = []
                            features_order = ["lag_1","lag_7","lag_14","ma7","ma14","weekday"]
                            for i in range(horizon_local):
                                row = [
                                    lag1,
                                    lag7,
                                    lag14,
                                    np.mean(recent[-7:]) if len(recent)>=1 else 0,
                                    np.mean(recent[-14:]) if len(recent)>=1 else 0,
                                    (int(last["Date_reported"].weekday()) + i + 1) % 7
                                ]
                                X = np.array(row).reshape(1,-1)
                                p = model_local.predict(X)[0]
                                p = max(0, p)
                                preds.append(p)
                                recent.append(p)
                                lag14, lag7, lag1 = lag7, lag1, p
                            return np.array(preds)

                        preds = iterative_forecast_local(model, df_feat, horizon_ml)
                        last_date = df[df["Country"]==country_ml]["Date_reported"].max()
                        future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(horizon_ml)]
                        figf = go.Figure()
                        hist = df[df["Country"]==country_ml].tail(90)
                        figf.add_trace(go.Scatter(x=hist["Date_reported"], y=hist["New_cases"], name="Actual", mode="lines"))
                        figf.add_trace(go.Scatter(x=future_dates, y=preds, name="Forecast", mode="lines+markers"))
                        figf.update_layout(title=f"Forecast ({model_name}) — {country_ml}", xaxis_title="Date", yaxis_title="New cases")
                        st.plotly_chart(figf, use_container_width=True)
                        out_df = pd.DataFrame({"Date":future_dates, model_name: preds.astype(int)})
                        st.dataframe(out_df)

# ===============================
# 6️⃣ Footer
# ===============================
st.markdown("""
---
👨‍💻 **Từ Nhật Anh** — Sinh viên CNTT, Đại học Sài Gòn  
📊 Dashboard phát triển bằng **Streamlit + Plotly + Pandas**  
Nguồn dữ liệu: [WHO COVID-19 Data Repository](https://covid19.who.int)
""")
