import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="WHO COVID-19 Dashboard", layout="wide")

st.title("🌍 WHO COVID-19 Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("data/latest.csv.gz")
    return df

df = load_data()

country = st.sidebar.selectbox("Chọn quốc gia", sorted(df["Country"].unique()))
filtered = df[df["Country"] == country]

st.metric("Tổng ca nhiễm", f"{int(filtered['Cumulative_cases'].values[0]):,}")
st.metric("Tổng ca tử vong", f"{int(filtered['Cumulative_deaths'].values[0]):,}")

fig = px.bar(filtered, x=["Cumulative_cases", "Cumulative_deaths"], 
             y=[filtered["Country"].values[0]]*2, orientation='h',
             labels={'x': 'Số lượng', 'y': ''},
             title=f"Tổng hợp COVID-19 tại {country}")
st.plotly_chart(fig, use_container_width=True)
