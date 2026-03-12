import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np

# Tiêu đề của trang web
st.title("Công cụ Chỉnh sửa Màu Ảnh AI 🎨")
st.write("Tải ảnh của bạn lên để hệ thống phân tích và chỉnh sửa!")

# Tạo nút Upload file
uploaded_file = st.file_uploader("Chọn một bức ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Đọc ảnh người dùng tải lên
    image = Image.open(uploaded_file)
    
    # Chia giao diện thành 2 cột để so sánh Trước/Sau cho đẹp
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ảnh gốc")
        st.image(image, use_container_width=True)

    # 2. Khu vực Giả lập AI & Điều chỉnh
    st.markdown("---")
    st.subheader("⚙️ Tùy chỉnh (AI sẽ tự động đánh giá phần này)")
    
    # Thanh trượt chỉnh độ sáng (Ví dụ demo)
    brightness_factor = st.slider("Độ sáng (Brightness)", 0.5, 2.0, 1.0)
    
    # Logic xử lý ảnh (Bạn sẽ nhúng model AI, OpenCV vào đây)
    enhancer = ImageEnhance.Brightness(image)
    edited_image = enhancer.enhance(brightness_factor)

    # 3. Hiển thị ảnh Kết quả
    with col2:
        st.subheader("Ảnh kết quả")
        st.image(edited_image, use_container_width=True)
