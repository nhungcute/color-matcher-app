import cv2
import numpy as np

def adjust_brightness_contrast(image_cv, brightness=0, contrast=0):
    """
    Điều chỉnh độ sáng và độ tương phản của ảnh trên toàn bộ khung hình.
    brightness/contrast range phổ biến: -127 đến +127 (hoặc -100 đến +100 trong thanh trượt app)
    """
    img = np.int16(image_cv)
    img = img * (contrast/127 + 1) - contrast + brightness
    img = np.clip(img, 0, 255)
    return np.uint8(img)

def adjust_temperature(image_cv, temperature=0):
    """
    temperature > 0: Tăng màu ấm (cam/vàng/đỏ)
    temperature < 0: Tăng màu lạnh (cyan/blue)
    Mức độ: -100 đến 100
    """
    img_float = image_cv.astype(np.float32)
    
    if temperature > 0:
        # Tăng đỏ, giảm xanh
        img_float[:, :, 2] += temperature # Red channel
        img_float[:, :, 0] -= temperature * 0.5 # Blue channel
    elif temperature < 0:
        # Tăng xanh, giảm đỏ
        img_float[:, :, 0] -= temperature # Blue channel (trừ số âm là cộng)
        img_float[:, :, 2] += temperature * 0.5 # Red channel
        
    img_float = np.clip(img_float, 0, 255)
    return img_float.astype(np.uint8)

def skin_whitening(image_cv, mask, brightness_increase=20, saturation_decrease=10):
    """
    Làm sáng/trắng da dựa trên mask. Tăng độ sáng (Lightness) và giảm nhẹ độ bão hòa (Saturation)
    trong không gian màu HLS.
    """
    # Chỉ xử lý nếu có mask
    if np.count_nonzero(mask) == 0:
        return image_cv.copy()
        
    # Chuyển sang HLS
    hls = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HLS).astype(np.float32)
    
    # hls[:, :, 1] = Lightness
    # hls[:, :, 2] = Saturation
    
    # Feather mask để không bị cứng góc
    blur_mask = cv2.GaussianBlur(mask, (21, 21), 0)
    alpha = blur_mask.astype(float) / 255.0
    
    # Tính lượng thay đổi Lightness và Saturation cho từng pixel theo map mask
    l_increase = alpha * brightness_increase
    s_decrease = alpha * saturation_decrease
    
    # Cập nhật và cắt giá trị về 0-255
    hls[:, :, 1] = np.clip(hls[:, :, 1] + l_increase, 0, 255)
    hls[:, :, 2] = np.clip(hls[:, :, 2] - s_decrease, 0, 255)
    
    hls = hls.astype(np.uint8)
    # Reconvert về BGR
    result_cv = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)

    return result_cv
