import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Import các module AI vừa xây dựng
from assessment import assess_image
from segmentation import create_skin_mask
from processing import adjust_brightness_contrast, adjust_temperature, skin_whitening

st.set_page_config(page_title="AI Photo Editor", layout="wide")

st.title("Công cụ Chỉnh sửa Màu Ảnh AI 🎨")
st.write("Tải ảnh của bạn lên để hệ thống phân tích và tự động đề xuất chỉnh sửa!")

# Tạo nút Upload file
uploaded_file = st.file_uploader("Chọn một bức ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Đọc ảnh gốc bằng PIL và convert sang OpenCV format
    image_pil = Image.open(uploaded_file)
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # Chia giao diện thành 2 cột để so sánh Trước/Sau
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Ảnh gốc")
        st.image(image_pil, use_container_width=True)
        
    st.markdown("---")
    
    # === MODULE 1: Đánh giá ảnh ===
    st.subheader("🔍 Phân tích AI (Image Assessment)")
    with st.spinner("Đang phân tích ảnh..."):
        assessment_res = assess_image(image_cv)
        
    # Xử lý an toàn output của recommendations
    recs = assessment_res.get('recommendations', [])
    if isinstance(recs, list) and recs:
        rec_text = ", ".join(recs)
    else:
        rec_text = "Tùy ý điều chỉnh"
        
    st.write(f"**Độ sáng:** {assessment_res.get('brightness', {}).get('status', 'Bình thường')}")
    st.write(f"**Màu sắc:** {assessment_res.get('color', {}).get('status', 'Bình thường')}")
    st.write(f"**💡 AI Đề xuất:** {rec_text}")
    
    st.markdown("---")
    st.subheader("⚙️ Tùy chỉnh Xử lý ảnh")
    
    # Khởi tạo giá trị slider mặc định dựa trên AI Assessment
    def_bright = 0
    if assessment_res.get('brightness', {}).get('status') == "Hơi tối":
        def_bright = 30
    elif assessment_res.get('brightness', {}).get('status') == "Thừa sáng":
        def_bright = -30
        
    def_temp = 0
    color_status = assessment_res.get('color', {}).get('status', '')
    if isinstance(color_status, str):
        if "Ám Xanh dương (Blue)" in color_status:
            def_temp = 20
        elif "Ám Đỏ" in color_status:
            def_temp = -20
        
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        obj_brightness = st.slider("Độ sáng (Brightness)", -100, 100, def_bright)
    with c2:
        obj_contrast = st.slider("Độ tương phản (Contrast)", -100, 100, 0)
    with c3:
        obj_temp = st.slider("Nhiệt độ màu (Temperature)", -100, 100, def_temp)
    with c4:
        obj_rotate = st.slider("Xoay ảnh (°)", -180, 180, 0)
        
    st.markdown("#### Tính năng Nâng cao (AI Segmentation)")
    enable_skin_whitening = st.checkbox("Làm sáng & Trắng da mặt bằng AI")
    
    # === MODULE 2 & 3: Phân vùng và Xử lý ===
    with st.spinner("Đang xử lý ảnh..."):
        # Bước 0: Xoay ảnh
        if obj_rotate != 0:
            (h, w) = image_cv.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, obj_rotate, 1.0)
            
            # Tính toán kích thước ảnh mới bao quanh vừa vặn để không bị cắt gút
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            
            image_cv = cv2.warpAffine(image_cv, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            
        # Bước 1: Chỉnh sửa cơ bản trên toàn khung hình
        processed_cv = adjust_brightness_contrast(image_cv.copy(), brightness=obj_brightness, contrast=obj_contrast)
        processed_cv = adjust_temperature(processed_cv, temperature=obj_temp)
        
        # Bước 2: Nhận diện da (Semantic Segmentation) và Làm trắng (Processing)
        if enable_skin_whitening:
            skin_mask = create_skin_mask(image_cv)
            # Áp dụng làm trắng trên ảnh đã chỉnh sửa phần cơ bản
            processed_cv = skin_whitening(processed_cv, skin_mask, brightness_increase=30, saturation_decrease=15)
            
            with st.expander("Xem vùng da mặt do AI nhận diện (Mask)"):
                st.image(skin_mask, caption="Mask Trắng là phần được xử lý", width=300)

    # Convert lại ảnh kết quả sang RGB để Streamlit hiển thị
    final_rgb = cv2.cvtColor(processed_cv, cv2.COLOR_BGR2RGB)
    
    # Hiển thị ảnh Kết quả ở cột 2
    with col2:
        st.subheader("✨ Ảnh kết quả sau AI")
        st.image(Image.fromarray(final_rgb), use_container_width=True)
