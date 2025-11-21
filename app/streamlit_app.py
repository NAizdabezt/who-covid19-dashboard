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
# TAB: 🔮 Dự báo & Backtesting (Machine Learning)
# ---------------------------

with st.expander("🔮 Dự báo & Backtesting (Machine Learning) — Mở/Đóng"):
    st.markdown("### ⚙ Cấu hình mô hình & Backtesting")

    # ==== CHỌN QUỐC GIA ====
    countries_all = sorted(df["Country"].unique())
    country_sel = st.selectbox(
        "Chọn quốc gia để dự báo:",
        countries_all,
        index=countries_all.index("Viet Nam") if "Viet Nam" in countries_all else 0
    )

    # ==== CHỌN MÔ HÌNH ====
    model_choice = st.multiselect(
        "Chọn mô hình (có thể chọn nhiều):",
        ["LinearRegression", "RandomForest", "XGBoost"],
        default=["LinearRegression", "RandomForest"]
    )

    # ==== HYPERPARAMS ====
    horizon = st.selectbox("Horizon dự báo (ngày):", [7, 14, 30], index=1)
    window_size = st.slider("Kích thước Window train mỗi fold (ngày):", 60, 365, 180, 30)
    max_folds = st.slider("Số fold tối đa cho backtesting:", 1, 12, 6)

    run_backtest = st.button("🔁 Chạy Backtesting (rolling-window)")
    run_forecast = st.button("📈 Huấn luyện & Dự báo tương lai")

    st.write("---")

    # Lấy dữ liệu theo quốc gia
    df_country = df[df["Country"] == country_sel].sort_values("Date_reported").reset_index(drop=True)
    st.markdown(
        f"**Dữ liệu sử dụng:** {country_sel} — {len(df_country)} dòng "
        f"(_{df_country['Date_reported'].min().date()} → {df_country['Date_reported'].max().date()}_)."
    )

    # Nếu dữ liệu quá ít
    if len(df_country) < 30:
        st.warning("Dữ liệu quá ít để huấn luyện — cần >= 30 dòng.")
        st.stop()

    # ---------------------------
    # 🎯 FEATURE ENGINEERING
    # ---------------------------
    def make_features(df_country, target_col="New_cases"):
        dfc = df_country.sort_values("Date_reported").copy()
        dfc.reset_index(drop=True, inplace=True)

        for lag in [1,7,14]:
            dfc[f"lag_{lag}"] = dfc[target_col].shift(lag)

        dfc["ma7"] = dfc[target_col].rolling(7).mean().shift(1)
        dfc["ma14"] = dfc[target_col].rolling(14).mean().shift(1)
        dfc["weekday"] = dfc["Date_reported"].dt.weekday

        return dfc.dropna()

    features = ["lag_1", "lag_7", "lag_14", "ma7", "ma14", "weekday"]
    df_feat = make_features(df_country)

    if st.checkbox("📌 Xem trước dữ liệu feature (10 dòng cuối):"):
        st.dataframe(df_feat[["Date_reported", "New_cases"] + features].tail(10), height=260)

    # ---------------------------
    # 🎯 MÔ HÌNH: Train
    # ---------------------------
    def fit_model(name, X_train, y_train):
        if name == "LinearRegression":
            return LinearRegression().fit(X_train, y_train)
        elif name == "RandomForest":
            return RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)
        elif name == "XGBoost":
            return xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0).fit(X_train, y_train)
        else:
            raise ValueError("Unknown model")

    # ---------------------------
    # 🎯 BACKTESTING
    # ---------------------------
    def backtest(df_country, models, window_days, horizon, max_folds):
        dfc = make_features(df_country)
        results = {m: [] for m in models}

        n = len(dfc)
        if n < window_days + horizon:
            return {"error": f"Dữ liệu không đủ cho window={window_days} & horizon={horizon}"}

        possible_ends = list(range(window_days, n - horizon))
        if len(possible_ends) > max_folds:
            train_points = np.linspace(possible_ends[0], possible_ends[-1], max_folds, dtype=int)
        else:
            train_points = possible_ends

        for train_end in train_points:
            train = dfc.iloc[train_end - window_days : train_end]
            test = dfc.iloc[train_end : train_end + horizon]

            X_train, y_train = train[features].values, train["New_cases"].values
            X_test, y_test   = test[features].values,  test["New_cases"].values

            for m in models:
                model = fit_model(m, X_train, y_train)
                preds = np.clip(model.predict(X_test), 0, None)

                rmse = np.sqrt(mean_squared_error(y_test, preds))
                mae = mean_absolute_error(y_test, preds)

                results[m].append({"rmse": rmse, "mae": mae})

        return results

    # ---------------------------
    # RUN BACKTESTING
    # ---------------------------
    if run_backtest:
        with st.spinner("⏳ Đang chạy Backtesting..."):
            res = backtest(df_country, model_choice, window_size, horizon, max_folds)

        st.success("✔ Backtesting hoàn tất!")

        for m in res:
            scores = res[m]
            st.write(f"### 📌 {m} — Trung bình {len(scores)} folds")
            rmse_avg = np.mean([s["rmse"] for s in scores])
            mae_avg = np.mean([s["mae"] for s in scores])

            st.write(f"- RMSE trung bình: **{rmse_avg:.2f}**")
            st.write(f"- MAE trung bình: **{mae_avg:.2f}**")

            fig_err = px.bar(
                pd.DataFrame(scores),
                y="rmse", title=f"RMSE các fold — {m}", color="rmse"
            )
            st.plotly_chart(fig_err, use_container_width=True)

    # ---------------------------
    # 🎯 DỰ BÁO TƯƠNG LAI (train full)
    # ---------------------------
    if run_forecast:
        X_full = df_feat[features].values
        y_full = df_feat["New_cases"].values

        future_predictions = {}
        last_row = df_feat.iloc[-1]

        with st.spinner("⏳ Đang huấn luyện và dự báo..."):

            for m in model_choice:
                model = fit_model(m, X_full, y_full)
                preds = []

                lag1 = last_row["lag_1"]
                lag7 = last_row["lag_7"]
                lag14 = last_row["lag_14"]
                recent = list(df_feat["New_cases"].iloc[-14:].values)

                for i in range(horizon):
                    weekday = (int(last_row["Date_reported"].weekday()) + i + 1) % 7

                    row = {
                        "lag_1": lag1,
                        "lag_7": lag7,
                        "lag_14": lag14,
                        "ma7": np.mean(recent[-7:]),
                        "ma14": np.mean(recent[-14:]),
                        "weekday": weekday,
                    }

                    X_new = np.array([row[f] for f in features]).reshape(1, -1)
                    pred = max(0, model.predict(X_new)[0])
                    preds.append(pred)

                    # cập nhật lags
                    recent.append(pred)
                    lag14, lag7, lag1 = lag7, lag1, pred

                future_predictions[m] = preds

        # Vẽ biểu đồ forecast
        last_date = df_country["Date_reported"].max()
        future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(horizon)]

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=df_country["Date_reported"].tail(100),
            y=df_country["New_cases"].tail(100),
            mode="lines+markers",
            name="Actual (last 100 days)"
        ))

        colors = {"LinearRegression": "#1f77b4", "RandomForest": "#ff7f0e", "XGBoost": "#2ca02c"}

        for m in future_predictions:
            fig_fc.add_trace(go.Scatter(
                x=future_dates,
                y=future_predictions[m],
                mode="lines+markers",
                name=f"Forecast — {m}",
                line=dict(color=colors.get(m, None))
            ))

        fig_fc.update_layout(
            title=f"📈 Dự báo số ca nhiễm — {country_sel}",
            xaxis_title="Ngày",
            yaxis_title="Ca nhiễm mới"
        )

        st.plotly_chart(fig_fc, use_container_width=True)

        # bảng forecast
        df_out = pd.DataFrame({"Date": future_dates})
        for m in future_predictions:
            df_out[m] = np.array(future_predictions[m]).astype(int)

        st.dataframe(df_out)

# ===============================
# 6️⃣ Footer
# ===============================
st.markdown("""
---
👨‍💻 **Từ Nhật Anh** — Sinh viên CNTT, Đại học Sài Gòn  
📊 Dashboard phát triển bằng **Streamlit + Plotly + Pandas**  
Nguồn dữ liệu: [WHO COVID-19 Data Repository](https://covid19.who.int)
""")
