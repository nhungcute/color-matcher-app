import cv2
import numpy as np

def analyze_brightness(image_cv):
    """
    Phân tích độ sáng của ảnh dựa trên Histogram của kênh V (Value) trong không gian HSV.
    """
    # Chuyển sang không gian màu HSV
    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    
    # Tính mean của kênh V (0-255)
    mean_v = np.mean(v_channel)
    
    brightness_status = "Bình thường"
    if mean_v < 85: # 255 / 3 = 85
        brightness_status = "Hơi tối"
    elif mean_v > 170: # 255 * 2/3 = 170
        brightness_status = "Thừa sáng"
        
    return {
        "mean_brightness": mean_v,
        "status": brightness_status
    }

def analyze_color_cast(image_cv):
    """
    Phân tích độ lệch màu (color cast) bằng cách tính trung bình các kênh R, G, B.
    """
    # Tính trung bình các kênh B, G, R
    mean_b = np.mean(image_cv[:, :, 0])
    mean_g = np.mean(image_cv[:, :, 1])
    mean_r = np.mean(image_cv[:, :, 2])
    
    # Tính giá trị trung bình tổng
    mean_gray = (mean_r + mean_g + mean_b) / 3.0
    
    # Giả định đơn giản: nếu một kênh lệch so với trung bình quá nhiều
    threshold = 20.0 # Ngưỡng độ lệch so với gray
    
    color_cast = "Bình thường"
    issues = []
    if mean_r - mean_gray > threshold:
        issues.append("Ám Đỏ")
    if mean_g - mean_gray > threshold:
        issues.append("Ám Xanh lá (Green)")
    if mean_b - mean_gray > threshold:
        issues.append("Ám Xanh dương (Blue)")
        
    if issues:
        color_cast = ", ".join(issues)
        
    return {
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "status": color_cast
    }

def assess_image(image_cv):
    """
    Hàm tổng hợp đánh giá ảnh.
    """
    brightness_res = analyze_brightness(image_cv)
    color_res = analyze_color_cast(image_cv)
    
    # Gợi ý đơn giản từ phân tích
    recommendations = []
    if brightness_res["status"] == "Hơi tối":
        recommendations.append("Tăng độ sáng")
    elif brightness_res["status"] == "Thừa sáng":
        recommendations.append("Giảm độ sáng")
        
    if "Ám Xanh dương (Blue)" in color_res["status"]:
        recommendations.append("Tăng màu ấm (Vàng)")
    elif "Ám Đỏ" in color_res["status"]:
        recommendations.append("Giảm màu ấm (Cyan/Blue)")
        
    return {
        "brightness": brightness_res,
        "color": color_res,
        "recommendations": recommendations if recommendations else ["Tất cả đều ổn, bạn có thể chỉnh tùy ý thích"]
    }
