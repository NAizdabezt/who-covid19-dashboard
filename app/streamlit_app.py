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
    df = pd.read_csv("data/df_clean.csv.gz")
    latest = pd.read_csv("data/latest.csv.gz")
    return df, latest

df, latest = load_data()

# ===============================
# 3️⃣ Sidebar – bộ lọc
# ===============================
st.sidebar.header("🎚️ Bộ lọc dữ liệu")
countries = sorted(df["Country"].unique())
selected_country = st.sidebar.selectbox("Chọn quốc gia", ["Toàn cầu"] + countries)
show_globe = st.sidebar.checkbox("Hiển thị bản đồ 3D (Globe)", value=True)

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
# 5️⃣ Phần nội dung chính
# ===============================

# --- Biểu đồ theo thời gian ---
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

# Vẽ bản đồ 2D thế giới
st.subheader("🗺️ Bản đồ 2D COVID-19 theo quốc gia")
# Gom tổng ca nhiễm theo quốc gia
country_cases = df.groupby("Country", as_index=False)["New_cases"].sum()

fig = px.choropleth(
    country_cases,
    locations="Country",
    locationmode="country names",
    color="New_cases",
    color_continuous_scale="Reds",
    title="🌍 Tổng số ca nhiễm COVID-19 theo quốc gia (2020–2023)",
    projection="natural earth"
)

fig.update_layout(
    geo=dict(showframe=False, showcoastlines=True),
    paper_bgcolor="#0E1117",
    font=dict(color="white", size=14),
    title_x=0.5
)

st.plotly_chart(fig, use_container_width=True)
# --- Globe 3D ---
if show_globe:
    st.subheader("🌍 Bản đồ nhiệt COVID-19 (Ca/1 triệu dân)")
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
        showcountries=True,
        showcoastlines=True,
        showocean=True,
        showland=True,
        landcolor="LightGreen",
        oceancolor="LightBlue"
    )

    fig_globe.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        height=600
    )

    st.plotly_chart(fig_globe, use_container_width=True)

# ===============================
# 6️⃣ Footer
# ===============================
st.markdown("""
---
👨‍💻 **Từ Nhật Anh** — Sinh viên CNTT, Đại học Sài Gòn  
📊 Dashboard phát triển bằng **Streamlit + Plotly + Pandas**  
Nguồn dữ liệu: [WHO COVID-19 Data Repository](https://covid19.who.int)
""")
