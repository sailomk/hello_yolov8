import cv2
from scipy.spatial import distance as dist

# สมมติว่าวัตถุอ้างอิงของเรา (เช่น เหรียญหรือการ์ด) กว้าง 2.0 cm
REFERENCE_WIDTH = 2.0 

def get_size(frame):
    # 1. เตรียมภาพ (Grayscale -> Blur -> Canny)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blur, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # 2. หา Contours
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pixels_per_metric = None

    for c in cnts:
        if cv2.contourArea(c) < 500: # กรอง Noise เล็กๆ ออก
            continue

        # วาดกล่องรอบวัตถุ
        rect = cv2.minAreaRect(c)
        (x, y), (w, h), angle = rect
        
        # กรณีวัตถุแรกคือวัตถุอ้างอิง (ควรวางไว้ซ้ายสุดของภาพ)
        if pixels_per_metric is None:
            pixels_per_metric = w / REFERENCE_WIDTH
        
        # คำนวณขนาดจริง
        actual_w = w / pixels_per_metric
        actual_h = h / pixels_per_metric

        # แสดงผลบนจอ
        cv2.putText(frame, f"{actual_w:.1f}x{actual_h:.1f}cm", (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    return frame

# รันกล้อง MacBook
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    result = get_size(frame)
    cv2.imshow("Measure Tool", result)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()