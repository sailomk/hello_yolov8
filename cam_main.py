import cv2
import requests
import numpy as np
from ultralytics import YOLO

# 1. ตั้งค่า URL (Android IP Webcam) และโหลด Model
url = "http://192.168.200.179:8080/video"
model = YOLO('yolov8n.pt') 

# 2. เริ่มดึง Stream แบบ MJPEG
stream = requests.get(url, stream=True)
bytes_data = bytes()

print("ระบบกำลังค้นหาเฉพาะ 'คน'... กด 'q' เพื่อหยุด")

for chunk in stream.iter_content(chunk_size=1024):
    bytes_data += chunk
    a = bytes_data.find(b'\xff\xd8')
    b = bytes_data.find(b'\xff\xd9')
    
    if a != -1 and b != -1:
        jpg = bytes_data[a:b+2]
        bytes_data = bytes_data[b+2:]
        
        # แปลง Byte เป็นภาพ
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        if frame is not None:
            # 3. Tracking เฉพาะคน (classes=0)
            # conf=0.5 คือค่าความเชื่อมั่น ถ้าต่ำกว่านี้จะไม่แสดงผล (ปรับได้ตามความเหมาะสม)
            results = model.track(frame, persist=True, classes=[0], conf=0.4, verbose=False)

            # 4. ตรวจสอบว่าพบคนหรือไม่ และวาดผลลัพธ์
            if results[0].boxes.id is not None:
                annotated_frame = results[0].plot()
                person_count = len(results[0].boxes)
            else:
                # ถ้าไม่พบคนเลย ให้ใช้ภาพต้นฉบับ
                annotated_frame = frame
                person_count = 0
            
            # 5. แสดงจำนวนคนที่นับได้บนหน้าจอ
            cv2.rectangle(annotated_frame, (10, 10), (250, 60), (0, 0, 0), -1) # แถบดำหลังตัวเลข
            cv2.putText(annotated_frame, f"Person Count: {person_count}", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow('Android Person Tracker', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()