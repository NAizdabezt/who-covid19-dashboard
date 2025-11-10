import streamlit as st
import pandas as pd
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

# Gán lại cho df chính
df = df_filtered.copy()


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
    "📋 Dữ liệu chi tiết"
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
    st.subheader("📊 Thống kê tổng quan")

    # Top 10 quốc gia có số ca cao nhất
    top_countries = latest.sort_values(by="Cumulative_cases", ascending=False).head(10)

    st.markdown("### 🌍 Top 10 quốc gia có tổng ca nhiễm cao nhất")

    fig_top = px.bar(
        top_countries.sort_values("Cumulative_cases", ascending=True),
        x="Cumulative_cases",
        y="Country",
        orientation="h",  # 👉 biểu đồ ngang
        text="Cumulative_cases",
        color="Cumulative_cases",
        color_continuous_scale="Reds",
        labels={"Cumulative_cases": "Tổng ca nhiễm", "Country": "Quốc gia"},
        title="Top 10 quốc gia có tổng ca nhiễm COVID-19 cao nhất",
    )

    fig_top.update_traces(
        texttemplate="%{text:,}",  # Hiển thị số có dấu phẩy
        textposition="outside",
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
        margin=dict(l=50, r=20, t=80, b=20)
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

# ===============================
# 6️⃣ Footer
# ===============================
st.markdown("""
---
👨‍💻 **Từ Nhật Anh** — Sinh viên CNTT, Đại học Sài Gòn  
📊 Dashboard phát triển bằng **Streamlit + Plotly + Pandas**  
Nguồn dữ liệu: [WHO COVID-19 Data Repository](https://covid19.who.int)
""")
