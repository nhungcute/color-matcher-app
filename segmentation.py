import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def get_face_landmarks(image_cv):
    """
    Sử dụng MediaPipe Face Mesh để lấy landmarks của khuôn mặt.
    """
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        # MediaPipe yêu cầu ảnh RGB
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None
            
        return results.multi_face_landmarks[0]

def create_skin_mask(image_cv):
    """
    Tạo mask (mặt nạ) cho vùng da mặt sử dụng MediaPipe và Skin Color Thresholding (YCbCr).
    """
    height, width, _ = image_cv.shape
    
    # Lấy Face Landmarks
    landmarks = get_face_landmarks(image_cv)
    
    # Mask ban đầu đen
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if landmarks:
        # Lấy tọa độ các điểm trên contour (đường viền) của khuôn mặt
        face_oval_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        points = []
        for idx in face_oval_indices:
            landmark = landmarks.landmark[idx]
            x, y = int(landmark.x * width), int(landmark.y * height)
            points.append((x, y))
            
        points = np.array(points, dtype=np.int32)
        # Tô vùng da mặt bằng trắng (255)
        cv2.fillPoly(mask, [points], (255))
        
        # TÙY CHỌN: Có thể lọc thêm một lần bằng YCrCb trên đoạn mask này để loại bỏ mắt/môi sâu hơn.
        # Nhưng ở MVP, ta lấy nguyên mảng face mesh
    else:
        # Fallback về YCrCb thresholding khi không thấy mặt rõ
        ycbcr = cv2.cvtColor(image_cv, cv2.COLOR_BGR2YCrCb)
        # Ngưỡng màu da chung chung
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycbcr, lower_skin, upper_skin)
        
    return mask
