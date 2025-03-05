import streamlit as st
from main_app.app import app
from main_app.model import init_model


# Kiểm tra xem mô hình đã được load chưa
if "model" not in st.session_state:
    weights_path = "resources/weights/nkf_epoch70.pt"   # Đường dẫn đến file trọng số mô hình
    st.session_state.model = init_model(weights_path)   # Load mô hình 1 lần duy nhất

if __name__ == "__main__":
    # Chạy ứng dụng Streamlit với mô hình đã khởi tạo
    app(st.session_state.model)
    

# Cách chạy: streamlit run main.py
# Mở trình duyệt và truy cập đến địa chỉ http://localhost:8501
