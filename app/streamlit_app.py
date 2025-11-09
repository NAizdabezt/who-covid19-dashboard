import streamlit as st
import pandas as pd

# Đọc dữ liệu đã nén (làm sạch sẵn)
df = pd.read_csv("data/df_clean.csv.gz")  # hoặc latest.csv.gz
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Đọc dữ liệu đã xử lý
df = pd.read_csv("data/latest.csv.gz")

# Vẽ globe COVID-19
fig = go.Figure(go.Choropleth(
    locations=df['Country_code3'],
    z=df['Cases_per_million'],
    text=df['Country'] + "<br>" +
         "Population: " + df['Population'].astype(str) + "<br>" +
         "Cumulative cases: " + df['Cumulative_cases'].astype(str),
    colorscale='Reds',
    colorbar_title='Ca/1 triệu dân',
    marker_line_color='black',
    marker_line_width=0.5
))

fig.update_geos(
    projection_type="orthographic",
    showcountries=True,
    showcoastlines=True,
    showocean=True,
    showland=True,
    landcolor="LightGreen",
    oceancolor="LightBlue",
)

fig.update_layout(
    title_text='🌍 Tỷ lệ ca COVID-19 trên 1 triệu dân theo quốc gia',
    margin={"r":0,"t":50,"l":0,"b":0}
)

# Hiển thị trong Streamlit
st.plotly_chart(fig, use_container_width=True)
