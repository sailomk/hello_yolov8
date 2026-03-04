from ultralytics import YOLO
import cv2

# 1. โหลด Model (ตัว 'n' คือ nano ขนาดเล็กและเร็วมาก)
model = YOLO('yolov8n.pt')

# 2. ระบุไฟล์ภาพที่ต้องการนับ
image_path = 'penq.jpg' 
results = model(image_path)

# 3. ดึงข้อมูลการตรวจจับ
for result in results:
    # นับจำนวนวัตถุทั้งหมดที่เจอ
    count = len(result.boxes)
    print(f"พบวัตถุทั้งหมด: {count} รายการ")

    # แสดงรายการว่าเจออะไรบ้างและจำนวนเท่าไหร่
    names = result.names
    detected_classes = result.boxes.cls.tolist()
    
    counts_dict = {}
    for class_id in detected_classes:
        label = names[int(class_id)]
        counts_dict[label] = counts_dict.get(label, 0) + 1
    
    print("รายละเอียดการนับ:", counts_dict)

    # 4. แสดงผลลัพธ์เป็นภาพ
    annotated_frame = result.plot()
    cv2.imshow("Object Counting", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()