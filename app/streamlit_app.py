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


# Checkbox hiển thị bản đồ
show_globe2d = st.sidebar.checkbox("🗺️ Hiển thị bản đồ 2D", value=True)
show_globe3d = st.sidebar.checkbox("🌐 Hiển thị bản đồ 3D", value=True)

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

# ⚠️ Không tính theo dân số vì không có cột Population
latest_filtered["Fatality_rate"] = (
    (latest_filtered["Cumulative_deaths"] / latest_filtered["Cumulative_cases"]) * 100
).round(2)
latest_filtered["Fatality_rate"] = latest_filtered["Fatality_rate"].fillna(0)

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
    if "Country_code" in latest_filtered.columns:
        fig = px.choropleth(
            latest_filtered,
            locations="Country_code",
            color="Cumulative_cases",
            hover_name="Country",
            color_continuous_scale="Reds",
            title="🌍 Tổng số ca nhiễm COVID-19 theo quốc gia (theo thời gian lọc)",
            projection="natural earth"
        )
        fig.update_layout(
            geo=dict(showframe=False, showcoastlines=True),
            paper_bgcolor="#0E1117",
            font=dict(color="white", size=14),
            title_x=0.5
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Không tìm thấy cột mã quốc gia để vẽ bản đồ.")

# --- TAB 3: Top quốc gia ---
with tab3:
    st.subheader("🏆 Top 10 quốc gia có tổng ca nhiễm cao nhất (theo thời gian lọc)")
    top10 = latest_filtered.nlargest(10, "Cumulative_cases")
    fig_top10 = px.bar(
        top10, x="Country", y="Cumulative_cases",
        color="Cumulative_cases", color_continuous_scale="Reds",
        labels={"Cumulative_cases": "Tổng ca nhiễm"},
        title="Top 10 quốc gia có tổng ca nhiễm cao nhất"
    )
    st.plotly_chart(fig_top10, use_container_width=True)

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
