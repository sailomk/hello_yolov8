import cv2
import requests
import numpy as np
from ultralytics import YOLO

# 1. ตั้งค่าพื้นฐาน
url = "http://192.168.200.68:8080/video"
model = YOLO('yolov8n.pt')

# กำหนดเส้นแนวตั้ง (Vertical Line): สมมติที่ x = 400 
# (ปรับตามความกว้างของภาพคุณ เช่น ถ้าภาพกว้าง 1280 อาจตั้งที่ 640)
line_x = 400 
total_count = 0
tracked_ids = set() 
prev_x_centers = {}

stream = requests.get(url, stream=True)
bytes_data = bytes()

print("เริ่มระบบนับคนผ่านเส้นแนวตั้ง... กด 'q' เพื่อหยุด")

for chunk in stream.iter_content(chunk_size=1024):
    bytes_data += chunk
    a = bytes_data.find(b'\xff\xd8')
    b = bytes_data.find(b'\xff\xd9')
    
    if a != -1 and b != -1:
        jpg = bytes_data[a:b+2]
        bytes_data = bytes_data[b+2:]
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        if frame is not None:
            # 2. Track เฉพาะคน
            results = model.track(frame, persist=True, classes=[0], verbose=False)

            # วาดเส้นนับแนวตั้ง (สีแดง)
            cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 0, 255), 3)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)

                for box, id in zip(boxes, ids):
                    x1, y1, x2, y2 = box
                    # ใช้จุดกึ่งกลางของตัวคน (Center X) เพื่อเช็คการข้ามเส้น
                    current_center_x = int((x1 + x2) / 2)
                    current_center_y = int((y1 + y2) / 2)

                    # ตรวจสอบการข้ามเส้นจากซ้ายไปขวา
                    if id in prev_x_centers:
                        old_x = prev_x_centers[id]
                        
                        # Logic: ถ้าเฟรมก่อนอยู่ซ้ายของเส้น และเฟรมนี้อยู่ขวาของเส้น
                        if old_x <= line_x and current_center_x > line_x:
                            if id not in tracked_ids:
                                total_count += 1
                                tracked_ids.add(id)
                                print(f"ข้ามเส้นจากซ้ายไปขวา! ID: {id} | รวม: {total_count}")
                        
                        # (Option) ถ้าต้องการนับจากขวาไปซ้ายด้วย ให้เพิ่มเงื่อนไขนี้:
                        # elif old_x >= line_x and current_center_x < line_x:
                        #     ...เพิ่ม count...

                    # อัปเดตตำแหน่งล่าสุด
                    prev_x_centers[id] = current_center_x
                    
                    # วาดจุดกึ่งกลางและ ID
                    cv2.circle(frame, (current_center_x, current_center_y), 5, (0, 255, 0), -1)
                    cv2.putText(frame, f"ID:{id}", (current_center_x, current_center_y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 3. แสดงผล Total Count
            cv2.rectangle(frame, (10, 10), (320, 70), (0, 0, 0), -1)
            cv2.putText(frame, f"TOTAL COUNT: {total_count}", (25, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

            cv2.imshow('Vertical Line Counter', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()