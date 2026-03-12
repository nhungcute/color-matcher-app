import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Import các module AI vừa xây dựng
from assessment import assess_image
from segmentation import create_skin_mask
from processing import adjust_brightness_contrast, adjust_temperature, skin_whitening

from streamlit_image_comparison import image_comparison

st.set_page_config(page_title="AI Photo Editor", layout="wide")

st.title("Công cụ Chỉnh sửa Màu Ảnh AI 🎨")
st.write("Tải ảnh của bạn lên để hệ thống phân tích và tự động đề xuất chỉnh sửa!")

# Tạo nút Upload file
uploaded_file = st.file_uploader("Chọn một bức ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Đọc ảnh gốc bằng PIL và convert sang OpenCV format
    image_pil = Image.open(uploaded_file)
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # Khởi tạo state
    if "rotation" not in st.session_state:
        st.session_state.rotation = 0
    if "zoom" not in st.session_state:
        st.session_state.zoom = 1.0
        
    # Thanh công cụ Top: Row chứa Tiêu đề, Nút Zoom và Nút xoay
    col_title, col_zoom_out, col_zoom_text, col_zoom_in, col_rotate = st.columns([0.5, 0.1, 0.1, 0.1, 0.2])
    
    with col_title:
        st.subheader("📸 Xem trước Ảnh")
        
    with col_zoom_out:
        if st.button("🔍 -", help="Thu nhỏ"):
            st.session_state.zoom = max(0.1, st.session_state.zoom - 0.2)
            
    with col_zoom_text:
        st.write(f"Zoom: {st.session_state.zoom:.1f}x")
        
    with col_zoom_in:
        if st.button("🔍 +", help="Phóng to"):
            st.session_state.zoom = min(10.0, st.session_state.zoom + 0.2)
            
    with col_rotate:
        if st.button("↻ Xoay 90°"):
            st.session_state.rotation = (st.session_state.rotation + 90) % 360
            
    # Quy chuẩn vòng xoay
    current_rot = st.session_state.rotation
    if current_rot > 180: current_rot -= 360
    
    # === Xử lý ảnh gốc ban đầu (Xoay) ===
    if current_rot != 0:
        (h, w) = image_cv.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, current_rot, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        image_cv = cv2.warpAffine(image_cv, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
    # Lưu lại bản sao của ảnh gốc chưa chỉnh màu (để cho vào slider comparison)
    original_for_comparison = image_cv.copy()
        
    st.markdown("---")
    
    # === MODULE 1: Đánh giá ảnh ===
    st.subheader("🔍 Phân tích AI (Image Assessment)")
    with st.spinner("Đang phân tích ảnh..."):
        assessment_res = assess_image(image_cv)
        
    recs = assessment_res.get('recommendations', [])
    rec_text = ", ".join(recs) if isinstance(recs, list) and recs else "Tùy ý điều chỉnh"
        
    st.write(f"**Độ sáng:** {assessment_res.get('brightness', {}).get('status', 'Bình thường')}")
    st.write(f"**Màu sắc:** {assessment_res.get('color', {}).get('status', 'Bình thường')}")
    st.write(f"**💡 AI Đề xuất:** {rec_text}")
    
    st.markdown("---")
    st.subheader("⚙️ Tùy chỉnh Xử lý ảnh")
    
    # Mặc định Slider AI
    def_bright = 30 if assessment_res.get('brightness', {}).get('status') == "Hơi tối" else \
                 -30 if assessment_res.get('brightness', {}).get('status') == "Thừa sáng" else 0
                 
    color_status = assessment_res.get('color', {}).get('status', '')
    def_temp = 20 if "Ám Xanh dương" in (color_status or "") else \
              -20 if "Ám Đỏ" in (color_status or "") else 0
        
    c1, c2, c3 = st.columns(3)
    with c1: obj_brightness = st.slider("Độ sáng (Brightness)", -100, 100, def_bright)
    with c2: obj_contrast = st.slider("Độ tương phản (Contrast)", -100, 100, 0)
    with c3: obj_temp = st.slider("Nhiệt độ màu (Temperature)", -100, 100, def_temp)
        
    st.markdown("#### Tính năng Nâng cao (AI Segmentation)")
    enable_skin_whitening = st.checkbox("Làm sáng & Trắng da mặt bằng AI")
    
    # === MODULE 2 & 3: Xử lý màu sắc và AI ===
    with st.spinner("Đang xử lý ảnh..."):
        processed_cv = adjust_brightness_contrast(image_cv.copy(), brightness=obj_brightness, contrast=obj_contrast)
        processed_cv = adjust_temperature(processed_cv, temperature=obj_temp)
        
        if enable_skin_whitening:
            skin_mask = create_skin_mask(image_cv)
            processed_cv = skin_whitening(processed_cv, skin_mask, brightness_increase=30, saturation_decrease=15)
            
    # Tính toán Zoom cho ảnh hiển thị
    current_zoom = st.session_state.zoom
    if current_zoom != 1.0:
        h, w = original_for_comparison.shape[:2]
        new_w, new_h = int(w * current_zoom), int(h * current_zoom)
        original_for_comparison = cv2.resize(original_for_comparison, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        processed_cv = cv2.resize(processed_cv, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Hiển thị Thanh trượt So sánh Trước/Sau
    st.markdown("### ✨ Kéo để Xem Trước & Sau")
    image_comparison(
        img1=Image.fromarray(cv2.cvtColor(original_for_comparison, cv2.COLOR_BGR2RGB)),
        img2=Image.fromarray(cv2.cvtColor(processed_cv, cv2.COLOR_BGR2RGB)),
        label1="Ảnh Gốc",
        label2="Sau khi chỉnh AI"
    )
