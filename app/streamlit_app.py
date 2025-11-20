import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
from math import sqrt

# ===============================
# 1️⃣ Cấu hình trang
# ===============================
st.set_page_config(
    page_title="WHO COVID-19 Global Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 WHO COVID-19 Global COVID-19 Dashboard")
st.markdown("""
### Tổng quan tình hình COVID-19 toàn cầu  
Dữ liệu cập nhật và trực quan hóa theo quốc gia từ **World Health Organization (WHO)**.  
""")

# ===============================
# 2️⃣ Đọc dữ liệu
# ===============================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/df_clean.csv.gz")
        latest = pd.read_csv("data/latest.csv.gz")
        df["Date_reported"] = pd.to_datetime(df["Date_reported"], errors="coerce")
        return df, latest
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu: {e}")
        return None, None

df, latest = load_data()

if df is None or latest is None:
    st.stop()  # Dừng app nếu chưa có dữ liệu

# ===============================
# 3️⃣ Sidebar – bộ lọc
# ===============================
st.sidebar.header("🎚️ Bộ lọc dữ liệu")

countries = sorted(df["Country"].unique())
selected_country = st.sidebar.selectbox("Chọn quốc gia", ["Toàn cầu"] + countries)

# Lấy khoảng ngày có trong dữ liệu
min_ts = pd.to_datetime(df["Date_reported"].min())
max_ts = pd.to_datetime(df["Date_reported"].max())
min_date = min_ts.date()
max_date = max_ts.date()

# Hiển thị chú thích
st.sidebar.caption(f"📅 Dữ liệu hiện có từ **{min_date}** đến **{max_date}**.")

# ===============================
# 🗓️ Bộ lọc theo thời gian (phiên bản an toàn tuyệt đối)
# ===============================

# Đảm bảo cột ngày là datetime
df["Date_reported"] = pd.to_datetime(df["Date_reported"], errors="coerce")

# Lấy mốc min/max
min_ts = df["Date_reported"].min()
max_ts = df["Date_reported"].max()


st.sidebar.subheader("📅 Khoảng thời gian")

# Người dùng chọn khoảng ngày
date_input = st.sidebar.date_input(
    "Chọn khoảng thời gian",
    value=(min_ts.date(), max_ts.date())
)

# ✅ Kiểm tra trường hợp click 1 ngày
if isinstance(date_input, (list, tuple)) and len(date_input) == 2:
    start_ts = pd.to_datetime(date_input[0])
    end_ts = pd.to_datetime(date_input[1])
elif isinstance(date_input, (list, tuple)) and len(date_input) == 1:
    # chỉ click 1 lần → bỏ qua, dùng full range
    start_ts, end_ts = min_ts, max_ts
else:
    # nếu streamlit trả về 1 giá trị scalar (click 1 ngày)
    start_ts, end_ts = min_ts, max_ts

# ✅ Đảm bảo hợp lệ trong range
if start_ts < min_ts:
    start_ts = min_ts
if end_ts > max_ts:
    end_ts = max_ts
if start_ts > end_ts:
    start_ts, end_ts = end_ts, start_ts

# ✅ Lọc dữ liệu
df_filtered = df[(df["Date_reported"] >= start_ts) & (df["Date_reported"] <= end_ts)]

st.caption(f"📆 Dữ liệu hiển thị: từ **{start_ts.date()}** đến **{end_ts.date()}**")

# # Cập nhật lại df chính
# df = df_filtered.copy()


# # Checkbox hiển thị bản đồ
# show_globe2d = st.sidebar.checkbox("🗺️ Hiển thị bản đồ 2D", value=True)
# show_globe3d = st.sidebar.checkbox("🌐 Hiển thị bản đồ 3D", value=True)

# Lọc dữ liệu chính bằng khoảng ngày mới
df = df_filtered.copy()

# ===============================
# ✅ Sau khi lọc theo thời gian xong
# ===============================
# Tạo bảng latest_filtered: tổng ca và tử vong trong khoảng đã lọc
latest_filtered = (
    df_filtered.groupby(["Country", "Country_code"], as_index=False)
    .agg({
        "New_cases": "sum",
        "New_deaths": "sum"
    })
    .rename(columns={
        "New_cases": "Cumulative_cases",
        "New_deaths": "Cumulative_deaths"
    })
)

# ✅ Ghép thêm thông tin bổ sung từ file latest gốc (đã có Country_code3, Population,…)
latest_filtered = latest_filtered.merge(
    latest[["Country", "Country_code", "Country_code3", "Population"]],
    on=["Country", "Country_code"],
    how="left"
)

# Tính thêm các chỉ số
latest_filtered["Cases_per_million"] = (
    latest_filtered["Cumulative_cases"] / (latest_filtered["Population"] / 1_000_000)
)
latest_filtered["Fatality_rate"] = (
    latest_filtered["Cumulative_deaths"] / latest_filtered["Cumulative_cases"]
) * 100

# ===============================
# 4️⃣ KPI Cards
# ===============================
col1, col2, col3, col4 = st.columns(4)
total_cases = latest_filtered["Cumulative_cases"].sum()
total_deaths = latest_filtered["Cumulative_deaths"].sum()
fatality_rate = total_deaths / total_cases * 100 if total_cases > 0 else 0
affected_countries = latest_filtered["Country"].nunique()

col1.metric("🦠 Tổng ca nhiễm", f"{total_cases:,}")
col2.metric("⚰️ Tổng ca tử vong", f"{total_deaths:,}")
col3.metric("📊 Tỷ lệ tử vong (%)", f"{fatality_rate:.2f}")
col4.metric("🌎 Quốc gia bị ảnh hưởng", f"{affected_countries}")

# ===============================
# 5️⃣ Tabs cho phần nội dung chính
# ===============================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Xu hướng ca nhiễm",
    "🗺️ Bản đồ thế giới",
    "🏆 Top quốc gia",
    "📋 Dữ liệu chi tiết",
])

# --- TAB 1: Xu hướng theo thời gian ---
with tab1:
    st.subheader("📈 Xu hướng ca nhiễm theo thời gian")
    if "selected_country" in locals() and selected_country != "Toàn cầu":
        country_data = df_filtered[df_filtered["Country"] == selected_country]
        fig_line = px.line(
            country_data, x="Date_reported", y="New_cases",
            title=f"Số ca nhiễm mới tại {selected_country}",
            labels={"Date_reported": "Ngày", "New_cases": "Ca nhiễm mới"},
            color_discrete_sequence=["#E74C3C"]
        )
    else:
        global_trend = df_filtered.groupby("Date_reported")[["New_cases", "New_deaths"]].sum().reset_index()
        fig_line = px.line(
            global_trend, x="Date_reported", y="New_cases",
            title="Số ca nhiễm mới toàn cầu theo thời gian",
            labels={"Date_reported": "Ngày", "New_cases": "Ca nhiễm mới"}
        )
    st.plotly_chart(fig_line, use_container_width=True)

# --- TAB 2: Bản đồ ---
with tab2:
    st.subheader("🗺️ Bản đồ COVID-19 theo quốc gia")

    # ✅ Bảo đảm có cột ISO3 từ dữ liệu gốc (latest)
    if "Country_code3" not in latest_filtered.columns:
        latest_filtered = latest_filtered.merge(
            latest[["Country", "Country_code3"]].drop_duplicates(),
            on="Country",
            how="left"
        )

    # --- Bộ chọn loại dữ liệu hiển thị ---
    map_metric = st.radio(
        "Chọn loại dữ liệu hiển thị:",
        ("Tổng số ca nhiễm", "Tỷ lệ ca/1 triệu dân"),
        horizontal=True,
    )

    color_col = (
        "Cases_per_million"
        if map_metric == "Tỷ lệ ca/1 triệu dân"
        else "Cumulative_cases"
    )
    color_title = "Ca/1 triệu dân" if color_col == "Cases_per_million" else "Ca nhiễm"

    # --- Bản đồ 2D ---
    st.markdown("#### 🗺️ Bản đồ 2D COVID-19 theo quốc gia")
    fig = px.choropleth(
        latest_filtered,
        locations="Country_code3",           # ISO3 code
        color=color_col,                     # chọn theo radio
        hover_name="Country",
        color_continuous_scale="Reds",
        title=f"🌍 {map_metric} theo quốc gia (2D)",
        projection="natural earth"
    )
    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True),
        paper_bgcolor="#0E1117",
        font=dict(color="white", size=14),
        title_x=0.5
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Bản đồ 3D ---
    st.markdown("#### 🌐 Bản đồ 3D (Interactive Globe)")
    fig_globe = go.Figure(go.Choropleth(
        locations=latest_filtered["Country_code3"],
        z=latest_filtered[color_col],
        text=(
            latest_filtered["Country"] + "<br>" +
            f"{color_title}: " + latest_filtered[color_col].round(2).astype(str)
        ),
        colorscale="Reds",
        colorbar_title=color_title,
        marker_line_color="black",
        marker_line_width=0.5
    ))

    fig_globe.update_geos(
        projection_type="orthographic",
        showcountries=True,
        showcoastlines=True,
        showocean=True,
        showland=True,
        landcolor="LightGreen",
        oceancolor="LightBlue",
        lataxis_showgrid=True,
        lonaxis_showgrid=True,
    )

    fig_globe.update_layout(
        title_text=f"{map_metric} theo quốc gia (Interactive Globe)",
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        height=600
    )

    st.plotly_chart(fig_globe, use_container_width=True)

# --- TAB 3: Tổng quan ---
with tab3:
    st.subheader("📊 Phân tích Top quốc gia COVID-19")

    # Thêm cột tỷ lệ tử vong (%)
    latest_filtered["Death_rate"] = (
        latest_filtered["Cumulative_deaths"] / latest_filtered["Cumulative_cases"].replace(0, None)
    ) * 100

    # Dropdown chọn loại thống kê
    option = st.selectbox(
        "Chọn loại thống kê hiển thị:",
        (
            "Tổng ca nhiễm cao nhất",
            "Tổng ca tử vong cao nhất",
            "Tỷ lệ tử vong cao nhất (%)",
            "Ca nhiễm trên 1 triệu dân cao nhất",
        )
    )

    # Xác định cột dữ liệu tương ứng
    if option == "Tổng ca nhiễm cao nhất":
        metric_col = "Cumulative_cases"
        title = "🌍 Top 10 quốc gia có tổng ca nhiễm COVID-19 cao nhất"
        color_scale = "Reds"
    elif option == "Tổng ca tử vong cao nhất":
        metric_col = "Cumulative_deaths"
        title = "⚰️ Top 10 quốc gia có tổng ca tử vong COVID-19 cao nhất"
        color_scale = "OrRd"
    elif option == "Tỷ lệ tử vong cao nhất (%)":
        metric_col = "Death_rate"
        title = "💀 Top 10 quốc gia có tỷ lệ tử vong cao nhất (%)"
        color_scale = "Peach"
    else:
        metric_col = "Cases_per_million"
        title = "🌎 Top 10 quốc gia có ca nhiễm trên 1 triệu dân cao nhất"
        color_scale = "Reds"
        
    latest_filtered[metric_col] = pd.to_numeric(latest_filtered[metric_col], errors="coerce")

    # Lấy top 10 quốc gia theo lựa chọn
    top_countries = latest_filtered.nlargest(10, metric_col)

    # --- Vẽ biểu đồ ---
    st.markdown(f"### {title}")

    fig_top = px.bar(
        top_countries.sort_values(metric_col, ascending=True),
        x=metric_col,
        y="Country",
        orientation="h",
        text=metric_col,
        color=metric_col,
        color_continuous_scale=color_scale,
        labels={metric_col: title, "Country": "Quốc gia"},
        title=title,
    )

    fig_top.update_traces(
        texttemplate="%{text:,.2f}" if "rate" in metric_col.lower() else "%{text:,}",
        textposition="outside", insidetextanchor="start", cliponaxis=False
    )


    fig_top.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        coloraxis_showscale=False,
        height=500,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="white", size=14),
        title=dict(x=0.5, font=dict(size=18)),
        margin=dict(l=50, r=80, t=80, b=20)
    )

    st.plotly_chart(fig_top, use_container_width=True)

# --- TAB 4: Dữ liệu chi tiết ---
with tab4:
    st.subheader("📋 Dữ liệu chi tiết theo quốc gia (theo thời gian lọc)")
    st.dataframe(
        latest_filtered[["Country", "Cumulative_cases", "Cumulative_deaths", "Fatality_rate"]]
        .sort_values(by="Cumulative_cases", ascending=False)
        .reset_index(drop=True)
    )   

# ---------------------------
# TAB: 🔮 Dự báo & Backtesting
# ---------------------------

# ==== HỖ TRỢ: tạo feature lag + rolling ====
def make_features(df_country, target_col="New_cases", max_lag=14):
    """
    Input: df_country có Date_reported (datetime) và target_col (New_cases)
    Output: DataFrame sorted, có lag1/7/14, ma7, ma14, weekday
    """
    dfc = df_country.sort_values("Date_reported").copy()
    dfc = dfc.reset_index(drop=True)
    # ensure index is continuous days? we will not reindex by date to avoid holes; assume daily reporting mostly present
    for lag in [1,7,14]:
        dfc[f"lag_{lag}"] = dfc[target_col].shift(lag)
    dfc["ma7"] = dfc[target_col].rolling(7, min_periods=1).mean().shift(1)
    dfc["ma14"] = dfc[target_col].rolling(14, min_periods=1).mean().shift(1)
    dfc["weekday"] = dfc["Date_reported"].dt.weekday
    dfc = dfc.dropna(subset=[f"lag_{l}" for l in [1,7,14]])  # remove top rows without lags
    return dfc

# ==== Huấn luyện model đơn giản (fit trên X,y) ====
def fit_model(name, X_train, y_train, random_state=42):
    if name == "LinearRegression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model
    elif name == "RandomForest":
        model = RandomForestRegressor(n_estimators=200, random_state=random_state)
        model.fit(X_train, y_train)
        return model
    elif name == "XGBoost":
        model = xgb.XGBRegressor(n_estimators=200, random_state=random_state, verbosity=0)
        model.fit(X_train, y_train)
        return model
    else:
        raise ValueError("Unknown model name")

# ==== Dự báo tương lai bằng cách lặp autoregressive với features lag/ma ====
def iterative_forecast(model, df_feat, horizon=14, features=None, target_col="New_cases"):
    """
    df_feat: dữ liệu đã có feature, sorted, dùng để khởi tạo last lags
    Trả về array dự báo length=horizon
    Strategy: lấy last row, dùng lag features để predict next, append, update lags/mas iteratively.
    """
    last = df_feat.iloc[-1].copy()
    preds = []
    # copy arrays of last values to update
    lag1 = last["lag_1"]
    lag7 = last["lag_7"]
    lag14 = last["lag_14"]
    # to compute moving averages we need queue of recent values
    recent = list(df_feat[target_col].iloc[-14:].values)  # at most last14
    for h in range(horizon):
        row = {}
        row["lag_1"] = lag1
        row["lag_7"] = lag7
        row["lag_14"] = lag14
        row["ma7"] = np.mean(recent[-7:]) if len(recent) >= 1 else np.mean(recent)
        row["ma14"] = np.mean(recent[-14:]) if len(recent) >= 1 else np.mean(recent)
        # weekday we approximate by incrementing day
        weekday = (int(last["Date_reported"].weekday()) + h + 1) % 7
        row["weekday"] = weekday
        X_row = np.array([row[f] for f in features]).reshape(1,-1)
        pred = model.predict(X_row)[0]
        if np.isnan(pred) or pred < 0:
            pred = max(0.0, 0.0)  # clamp to zero
        preds.append(pred)
        # update lags and recent
        recent.append(pred)
        lag14 = lag7 if isinstance(lag7, (int,float,np.number)) else lag7
        lag7 = lag1 if isinstance(lag1, (int,float,np.number)) else lag1
        lag1 = pred
    return np.array(preds)

# ==== Backtesting rolling-window ====
def backtest_models(df_country, model_names, window_size_days=180, horizon=14, max_folds=8, features=None, target_col="New_cases"):
    """
    df_country: per-country timeline with Date_reported and New_cases
    window_size_days: số ngày dùng để train mỗi fold
    horizon: forecast horizon per fold (7,14,30)
    max_folds: giới hạn fold để tránh quá nặng
    returns: result dict per model: metrics list and average
    """
    dfc = make_features(df_country, target_col=target_col)
    results = {m: [] for m in model_names}
    n = len(dfc)
    # compute fold starts: use last portion of df so test windows are near end
    # we will create folds end positions such that test_end <= last_index
    # start indices (train_end) will be evenly spaced
    if n < window_size_days + horizon:
        return {m: {"error": f"Không đủ dữ liệu cho window={window_size_days} và horizon={horizon} (n={n})"} for m in model_names}
    # possible train_end indices: from window_size_days to n - horizon
    possible_ends = list(range(window_size_days, n - horizon + 1))
    # choose up to max_folds evenly spaced
    if len(possible_ends) > max_folds:
        idxs = np.linspace(possible_ends[0], possible_ends[-1], max_folds, dtype=int)
    else:
        idxs = possible_ends
    for train_end in idxs:
        train_df = dfc.iloc[train_end - window_size_days: train_end]
        test_df = dfc.iloc[train_end: train_end + horizon]
        X_train = train_df[features].values
        y_train = train_df[target_col].values
        X_test = test_df[features].values
        y_test = test_df[target_col].values
        for m in model_names:
            model = fit_model(m, X_train, y_train)
            preds = model.predict(X_test)
            # clamp negatives
            preds = np.where(np.isnan(preds), 0.0, preds)
            preds = np.clip(preds, 0, None)
            rmse = sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            # MAPE safe
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.mean(np.abs((y_test - preds) / np.where(y_test==0, np.nan, y_test))) * 100
                if np.isnan(mape):
                    mape = np.nan
            results[m].append({"rmse": rmse, "mae": mae, "mape": mape})
    # aggregate
    summary = {}
    for m in model_names:
        if isinstance(results[m], dict) and "error" in results[m]:
            summary[m] = results[m]
            continue
        rmses = [r["rmse"] for r in results[m]]
        maes = [r["mae"] for r in results[m]]
        mapes = [r["mape"] for r in results[m] if not np.isnan(r["mape"])]
        summary[m] = {
            "folds": len(results[m]),
            "rmse_mean": float(np.mean(rmses)) if len(rmses)>0 else None,
            "mae_mean": float(np.mean(maes)) if len(maes)>0 else None,
            "mape_mean": float(np.mean(mapes)) if len(mapes)>0 else None,
            "per_fold": results[m]
        }
    return summary

# ====== Streamlit UI for ML tab ======
tab_ml = st.tab if False else None  # placeholder if you want to insert differently

# Create actual tab inside app tabs list
# If you already have `tab1, tab2, ... = st.tabs([...])`, append this tab accordingly.
# Below we create a standalone tab using st.expander if tabs structure differs.
with st.expander("🔮 Dự báo & Backtesting (Machine Learning) — Mở/Đóng"):
    st.markdown("### Cấu hình model & backtest")
    countries_all = sorted(df["Country"].unique())
    country_sel = st.selectbox("Chọn quốc gia để dự báo", countries_all, index=countries_all.index("Viet Nam") if "Viet Nam" in countries_all else 0)
    model_choice = st.multiselect("Chọn mô hình (có thể chọn nhiều)", ["LinearRegression","RandomForest","XGBoost"], default=["LinearRegression","RandomForest","XGBoost"])
    horizon = st.selectbox("Horizon (số ngày dự báo)", [7,14,30], index=1)
    window_size = st.slider("Window train cho mỗi fold (ngày)", min_value=60, max_value=365, value=180, step=30)
    max_folds = st.slider("Số fold tối đa cho backtesting (giảm để nhanh hơn)", min_value=1, max_value=12, value=6)
    run_backtest = st.button("🔁 Chạy Backtesting (rolling-window)")

    # chọn kiểu dự báo cuối cùng (train on full history)
    run_forecast = st.button("📈 Huấn luyện toàn bộ & Dự báo tương lai (train full)")

    st.write("---")
    # prepare data of selected country
    df_country = df[df["Country"] == country_sel].sort_values("Date_reported").reset_index(drop=True)
    st.markdown(f"**Dữ liệu chọn:** {country_sel} — {len(df_country)} dòng ({df_country['Date_reported'].min().date()} → {df_country['Date_reported'].max().date()})")
    if len(df_country) < 30:
        st.warning("Dữ liệu quá ít để huấn luyện/kiểm định: cần >= 30 dòng.")

    features = ["lag_1","lag_7","lag_14","ma7","ma14","weekday"]
    df_feat = make_features(df_country)
    # show last few rows of features
    if st.checkbox("Hiện preview features (last 5 dòng)"):
        st.dataframe(df_feat[["Date_reported","New_cases"] + features].tail(10))

    # ---- Backtesting ----
    if run_backtest:
        with st.spinner("Đang chạy backtesting... (có thể mất vài chục giây tuỳ cấu hình)"):
            summary = backtest_models(df_country, model_choice, window_size_days=window_size, horizon=horizon, max_folds=max_folds, features=features)
        st.success("Hoàn tất backtesting")
        # hiển thị bảng tóm tắt
        rows = []
        for m in model_choice:
            val = summary.get(m)
            if val is None:
                continue
            if "error" in val:
                st.error(f"{m}: {val['error']}")
                continue
            rows.append({
                "model": m,
                "folds": val["folds"],
                "rmse_mean": round(val["rmse_mean"],2) if val["rmse_mean"] is not None else None,
                "mae_mean": round(val["mae_mean"],2) if val["mae_mean"] is not None else None,
                "mape_mean": round(val["mape_mean"],2) if val["mape_mean"] is not None else None
            })
        if len(rows) > 0:
            st.table(pd.DataFrame(rows).sort_values("rmse_mean"))
            # bar chart rmse compare
            df_comp = pd.DataFrame(rows)
            fig_err = px.bar(df_comp, x="model", y="rmse_mean", title=f"RMSE trung bình (horizon={horizon}d)", text="rmse_mean")
            st.plotly_chart(fig_err, use_container_width=True)

    # ---- Train full + forecast ----
    if run_forecast:
        if len(df_feat) < 10:
            st.error("Không đủ dữ liệu để train full model.")
        else:
            st.info("Huấn luyện trên toàn bộ lịch sử (đã tạo feature) và dự báo tương lai.")
            X_full = df_feat[features].values
            y_full = df_feat["New_cases"].values
            forecasts = {}
            for m in model_choice:
                model = fit_model(m, X_full, y_full)
                preds = iterative_forecast(model, df_feat, horizon=horizon, features=features)
                forecasts[m] = preds
            # build plot: show history last 120 days + forecasts
            lookback = 120
            hist = df_country.tail(lookback).copy()
            hist_idx = hist["Date_reported"].tolist()
            last_date = df_country["Date_reported"].max()
            future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(horizon)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist["Date_reported"], y=hist["New_cases"], mode="lines+markers", name="Actual (last 120d)"))
            colors = {"LinearRegression":"#1f77b4","RandomForest":"#ff7f0e","XGBoost":"#2ca02c"}
            for m, preds in forecasts.items():
                fig.add_trace(go.Scatter(x=future_dates, y=preds, mode="lines+markers", name=f"Forecast: {m}", line=dict(color=colors.get(m,None))))
            fig.update_layout(title=f"Actual vs Forecast ({country_sel}) — horizon={horizon} days", xaxis_title="Date", yaxis_title="New cases")
            st.plotly_chart(fig, use_container_width=True)
            # show numeric table of forecasts
            df_f = pd.DataFrame({"Date": future_dates})
            for m, preds in forecasts.items():
                df_f[m] = preds.astype(int)
            st.dataframe(df_f)

    st.markdown("### Ghi chú")
    st.markdown("""
    - Backtesting dùng `rolling window` giống mô tả trong báo cáo (chạy nhiều fold, báo RMSE/MAE/ MAPE trung bình).  
    - Forecast (train full) huấn luyện trên toàn bộ dữ liệu hiện có (sau feature) rồi **dự báo autoregressive** (dùng lag/ma).  
    - Để nhanh, giảm `max_folds` hoặc `window_size` khi chạy trên Streamlit Cloud.  
    - Nếu muốn mình mở rộng: thêm CI (confidence interval) cho XGBoost bằng bootstrap, thêm gridsearch cho hyperparams, hoặc lưu model đã train vào cache để forecast nhanh.
    """)


# ===============================
# 6️⃣ Footer
# ===============================
st.markdown("""
---
👨‍💻 **Từ Nhật Anh** — Sinh viên CNTT, Đại học Sài Gòn  
📊 Dashboard phát triển bằng **Streamlit + Plotly + Pandas**  
Nguồn dữ liệu: [WHO COVID-19 Data Repository](https://covid19.who.int)
""")
