# 🌍 WHO COVID-19 Data Dashboard

Phân tích dữ liệu COVID-19 toàn cầu dựa trên nguồn dữ liệu chính thức từ **Tổ chức Y tế Thế giới (WHO)**.  
Dự án này được xây dựng nhằm trực quan hóa tình hình dịch bệnh theo **quốc gia, khu vực và thời gian**, đồng thời giúp người dùng theo dõi các chỉ số như:
- Số ca mắc và tử vong tích lũy
- Tỷ lệ tử vong (%)
- Số ca trung bình theo ngày
- So sánh theo khu vực (Region)

---

## 🗂️ Cấu trúc thư mục

who-covid19-dashboard/
│
├── data/
│ ├── raw/
│ │ └── WHO-COVID-19-global-daily-data.csv # Dữ liệu gốc từ WHO
│ ├── df_clean.csv.gz # Dữ liệu đã làm sạch
│ └── latest.csv.gz # Dữ liệu tổng hợp theo quốc gia
│
├── notebooks/
│ └── analysis.ipynb # Phân tích và trực quan hóa (Google Colab)
│
├── app/
│ └── streamlit_app.py # Dashboard Streamlit
│
├── requirements.txt # Thư viện cần thiết
└── README.md

yaml
Sao chép mã

---

## ⚙️ Cách chạy phân tích trên Google Colab

1. Mở notebook `notebooks/analysis.ipynb`
2. Tải dữ liệu từ thư mục `data/`
3. Chạy từng cell để xem kết quả phân tích và biểu đồ

---

## 🚀 Chạy dashboard trên Streamlit Cloud

1. Truy cập [Streamlit Cloud](https://share.streamlit.io/)
2. Kết nối với GitHub của bạn
3. Chọn repository này
4. Đặt **file chính** là:  
app/streamlit_app.py

yaml
Sao chép mã
5. Deploy ✅

---

## 🧠 Phân tích chính

- Phân tích số ca & tử vong theo **thời gian**
- So sánh giữa các **WHO Region**
- Trực quan hóa bản đồ toàn cầu
- Tỷ lệ tử vong và ca bệnh trên 1 triệu dân

---

## 🧩 Công nghệ sử dụng

- **Python 3.10+**
- **pandas**, **plotly**, **pycountry**, **fuzzywuzzy**
- **Streamlit** (triển khai dashboard)
- **Google Colab** (phân tích dữ liệu)

---

## 📊 Nguồn dữ liệu

- [WHO COVID-19 Global Data](https://covid19.who.int/data)
- [World Bank Population Dataset](https://datahub.io/core/population)

---

## 👨‍💻 Nhóm thực hiện

- Từ Nhật Anh  
- (Cập nhật thêm thành viên khác nếu có)

---

## 📝 Giấy phép

MIT License — sử dụng tự do cho học tập và nghiên cứu.
