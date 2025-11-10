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
# 🗓️ Bộ lọc theo thời gian – chống lỗi khi chọn 1 ngày
# ===============================
date_input = st.sidebar.date_input(
    "Chọn khoảng thời gian",
    value=[min_date, max_date]
)

# 🔧 Đảm bảo luôn có start_date và end_date
if isinstance(date_input, list) and len(date_input) == 2:
    start_date, end_date = date_input
else:
    start_date = date_input
    end_date = date_input  # nếu chọn 1 ngày, dùng cùng ngày cho start & end

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Giới hạn trong khoảng dữ liệu
start_date = max(start_date, min_date)
end_date = min(end_date, max_date)

# ✅ Lọc dữ liệu an toàn
df_filtered = df[
    (df["Date_reported"] >= start_date) &
    (df["Date_reported"] <= end_date)
]

st.caption(f"Hiển thị dữ liệu từ **{start_date.date()}** đến **{end_date.date()}**")

# Checkbox hiển thị bản đồ
show_globe2d = st.sidebar.checkbox("🗺️ Hiển thị bản đồ 2D", value=True)
show_globe3d = st.sidebar.checkbox("🌐 Hiển thị bản đồ 3D", value=True)

# Lọc dữ liệu theo ngày
df = df[(df["Date_reported"] >= pd.Timestamp(start_date)) & (df["Date_reported"] <= pd.Timestamp(end_date))]

# ===============================
# 4️⃣ KPI Cards
# ===============================
col1, col2, col3, col4 = st.columns(4)
total_cases = latest["Cumulative_cases"].sum()
total_deaths = latest["Cumulative_deaths"].sum()
fatality_rate = total_deaths / total_cases * 100
affected_countries = latest["Country"].nunique()

col1.metric("🦠 Tổng ca nhiễm", f"{total_cases:,}")
col2.metric("⚰️ Tổng ca tử vong", f"{total_deaths:,}")
col3.metric("📊 Tỷ lệ tử vong (%)", f"{fatality_rate:.2f}")
col4.metric("🌎 Quốc gia bị ảnh hưởng", f"{affected_countries}")

# ===============================
# 5️⃣ Tabs cho phần nội dung chính
# ===============================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Xu hướng ca nhiễm", "🗺️ Bản đồ thế giới", "🏆 Top quốc gia", "📋 Dữ liệu chi tiết"])

# --- TAB 1: Xu hướng theo thời gian ---
with tab1:
    st.subheader("📈 Xu hướng ca nhiễm theo thời gian")
    if selected_country == "Toàn cầu":
        global_trend = df.groupby("Date_reported")[["New_cases", "New_deaths"]].sum().reset_index()
        fig_line = px.line(global_trend, x="Date_reported", y="New_cases",
                           title="Số ca nhiễm mới toàn cầu theo thời gian",
                           labels={"Date_reported": "Ngày", "New_cases": "Ca nhiễm mới"})
    else:
        country_data = df[df["Country"] == selected_country]
        fig_line = px.line(country_data, x="Date_reported", y="New_cases",
                           title=f"Số ca nhiễm mới tại {selected_country}",
                           labels={"Date_reported": "Ngày", "New_cases": "Ca nhiễm mới"},
                           color_discrete_sequence=["#E74C3C"])
    st.plotly_chart(fig_line, use_container_width=True)

# --- TAB 2: Bản đồ ---
with tab2:
    if show_globe2d:
        st.subheader("🗺️ Bản đồ 2D COVID-19 theo quốc gia")
        country_cases = df.groupby("Country", as_index=False)["New_cases"].sum()
        fig = px.choropleth(
            country_cases,
            locations="Country",
            locationmode="country names",
            color="New_cases",
            color_continuous_scale="Reds",
            title="🌍 Tổng số ca nhiễm COVID-19 theo quốc gia",
            projection="natural earth"
        )
        fig.update_layout(
            geo=dict(showframe=False, showcoastlines=True),
            paper_bgcolor="#0E1117",
            font=dict(color="white", size=14),
            title_x=0.5
        )
        st.plotly_chart(fig, use_container_width=True)

    if show_globe3d:
        st.subheader("🌐 Bản đồ 3D (Globe)")
        fig_globe = go.Figure(go.Choropleth(
            locations=latest['Country_code3'],
            z=latest['Cases_per_million'],
            text=latest['Country'] + "<br>" +
                "Dân số: " + latest['Population'].astype(str) + "<br>" +
                "Tổng ca nhiễm: " + latest['Cumulative_cases'].astype(str),
            colorscale='Reds',
            colorbar_title='Ca/1 triệu dân',
            marker_line_color='black',
            marker_line_width=0.5
        ))
        fig_globe.update_geos(
            projection_type="orthographic",
            showcountries=True, showcoastlines=True,
            showocean=True, showland=True,
            landcolor="LightGreen", oceancolor="LightBlue"
        )
        fig_globe.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=600)
        st.plotly_chart(fig_globe, use_container_width=True)

# --- TAB 3: Top quốc gia ---
with tab3:
    st.subheader("🏆 Top 10 quốc gia có tổng ca nhiễm cao nhất")
    top10 = latest.nlargest(10, "Cumulative_cases")
    fig_top10 = px.bar(top10, x="Country", y="Cumulative_cases",
                       color="Cumulative_cases", color_continuous_scale="Reds",
                       labels={"Cumulative_cases": "Tổng ca nhiễm"},
                       title="Top 10 quốc gia có tổng ca nhiễm cao nhất")
    st.plotly_chart(fig_top10, use_container_width=True)

# --- TAB 4: Dữ liệu chi tiết ---
with tab4:
    st.subheader("📋 Dữ liệu chi tiết theo quốc gia")
    st.dataframe(
        latest[["Country", "Cumulative_cases", "Cumulative_deaths", "Cases_per_million", "Fatality_rate"]]
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
