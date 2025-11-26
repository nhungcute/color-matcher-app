import streamlit as st
import numpy as np
from PIL import Image
from skimage import exposure

def match_color(target_img, ref_img):
    # Chuyển đổi ảnh sang dạng mảng số (numpy array)
    target = np.array(target_img)
    ref = np.array(ref_img)

    # Áp dụng histogram matching
    # channel_axis=-1 để xử lý đa kênh màu (RGB)
    matched = exposure.match_histograms(target, ref, channel_axis=-1)
    
    return Image.fromarray(matched.astype('uint8'))

# --- GIAO DIỆN WEB ---
st.title("🎨 Tool Copy Màu Ảnh (Color Transfer)")
st.write("Upload ảnh mẫu và ảnh cần chỉnh để đồng bộ màu sắc.")

# Cột trái: Upload ảnh mẫu (Reference)
col1, col2 = st.columns(2)
with col1:
    st.header("1. Ảnh Mẫu (Lấy màu)")
    ref_file = st.file_uploader("Chọn ảnh mẫu", type=['jpg', 'png', 'jpeg'], key="ref")
    if ref_file:
        ref_image = Image.open(ref_file)
        st.image(ref_image, caption="Ảnh mẫu", use_container_width=True)

# Cột phải: Upload ảnh cần chỉnh (Target) - Hỗ trợ nhiều ảnh
with col2:
    st.header("2. Ảnh Cần Chỉnh")
    target_files = st.file_uploader("Chọn ảnh cần chỉnh (có thể chọn nhiều)", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True, key="target")

# Nút xử lý
if st.button("🚀 Thực hiện Copy Màu") and ref_file and target_files:
    st.divider()
    st.subheader("Kết quả")
    
    for t_file in target_files:
        target_image = Image.open(t_file)
        
        # Xử lý màu
        result_image = match_color(target_image, ref_image)
        
        # Hiển thị kết quả so sánh
        c1, c2 = st.columns(2)
        c1.image(target_image, caption="Gốc", use_container_width=True)
        c2.image(result_image, caption="Đã chỉnh màu", use_container_width=True)
        
        # Nút download (cần chuyển ảnh thành bytes để tải về)
        from io import BytesIO
        buf = BytesIO()
        result_image.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        
        st.download_button(
            label=f"⬇️ Tải ảnh {t_file.name} về",
            data=byte_im,
            file_name=f"edited_{t_file.name}",
            mime="image/jpeg"
        )
        st.divider()