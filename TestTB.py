import cv2
import numpy as np
import urllib.request
import time
from collections import defaultdict
from ultralytics import YOLO
import threading
from queue import Queue
import torch
import warnings
warnings.filterwarnings('ignore')

class MJPEGStreamReader:
    """คลาสสำหรับอ่าน MJPEG stream จากกล้อง IP"""
    
    def __init__(self, url, queue_size=5):
        self.url = url
        self.stream = None
        self.running = False
        self.frame_queue = Queue(maxsize=queue_size)
        self.thread = None
        self.bytes_data = b''
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.connection_attempts = 0
        self.max_attempts = 5
        
    def start(self):
        """เริ่มอ่าน stream ใน thread แยก"""
        self.running = True
        self.thread = threading.Thread(target=self._read_stream, daemon=True)
        self.thread.start()
        print("📡 เริ่มอ่าน MJPEG stream...")
        return self
    
    def stop(self):
        """หยุดการอ่าน stream"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.stream:
            try:
                self.stream.close()
            except:
                pass
        print("📡 หยุดการอ่าน stream")
    
    def _read_stream(self):
        """อ่านข้อมูล MJPEG stream และแยกเฟรม"""
        while self.running:
            try:
                req = urllib.request.Request(
                    self.url,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Accept': '*/*'
                    }
                )
                
                self.stream = urllib.request.urlopen(req, timeout=15)
                self.bytes_data = b''
                self.connection_attempts = 0
                
                print("✅ เชื่อมต่อ MJPEG stream สำเร็จ")
                
                while self.running:
                    try:
                        chunk = self.stream.read(4096)
                        if not chunk:
                            break
                            
                        self.bytes_data += chunk
                        
                        # ค้นหา JPEG markers
                        start_marker = self.bytes_data.find(b'\xff\xd8')
                        end_marker = self.bytes_data.find(b'\xff\xd9')
                        
                        if start_marker != -1 and end_marker != -1 and end_marker > start_marker:
                            jpg_data = self.bytes_data[start_marker:end_marker+2]
                            self.bytes_data = self.bytes_data[end_marker+2:]
                            
                            try:
                                frame_array = np.frombuffer(jpg_data, dtype=np.uint8)
                                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                                
                                if frame is not None and frame.size > 0:
                                    # อัปเดต FPS
                                    self.frame_count += 1
                                    current_time = time.time()
                                    if current_time - self.last_time >= 1.0:
                                        self.fps = self.frame_count
                                        self.frame_count = 0
                                        self.last_time = current_time
                                    
                                    # ใส่ frame ใน queue
                                    if self.frame_queue.full():
                                        try:
                                            self.frame_queue.get_nowait()
                                        except:
                                            pass
                                    self.frame_queue.put(frame)
                                    
                            except Exception as e:
                                continue
                                
                    except Exception as e:
                        break
                        
            except Exception as e:
                self.connection_attempts += 1
                if self.connection_attempts >= self.max_attempts:
                    print("❌ ไม่สามารถเชื่อมต่อได้หลังจากพยายามหลายครั้ง")
                    self.running = False
                    break
                time.sleep(2)
            
            if self.stream:
                try:
                    self.stream.close()
                except:
                    pass
                self.stream = None
            
            if self.running:
                print("🔄 กำลังลองเชื่อมต่อใหม่...")
                time.sleep(1)
    
    def get_frame(self):
        """ดึงเฟรมล่าสุดจาก queue"""
        try:
            if not self.frame_queue.empty():
                return self.frame_queue.get_nowait()
        except:
            pass
        return None

class ThaiCoinDetector:
    def __init__(self, model_path=None, use_simulation=True):
        """
        กำหนดค่าเริ่มต้นสำหรับการตรวจจับเหรียญบาทไทย
        
        Args:
            model_path: พาธไปยังโมเดลที่เทรนมาเฉพาะ
            use_simulation: ใช้โหมดจำลองการตรวจจับ (สำหรับทดสอบ)
        """
        self.use_simulation = use_simulation
        self.device = self._get_optimal_device()
        print(f"💻 ใช้ device: {self.device}")
        
        # ถ้าใช้โหมดจำลอง ไม่ต้องโหลดโมเดลจริง
        if use_simulation:
            print("🎮 ใช้โหมดจำลองการตรวจจับ (สำหรับทดสอบ)")
            self.model = None
        else:
            # เลือกใช้โมเดล YOLO
            try:
                if model_path and model_path != "":
                    print(f"⚙️ กำลังโหลดโมเดล: {model_path}")
                    self.model = YOLO(model_path)
                else:
                    print("⚙️ กำลังโหลดโมเดลพื้นฐาน YOLOv8n...")
                    self.model = YOLO('yolov8n.pt')
                
                # ย้ายโมเดลไปยัง device
                self.model.to(self.device)
                if hasattr(self.model, 'model'):
                    self.model.model.half = False
            except Exception as e:
                print(f"⚠️ ไม่สามารถโหลดโมเดลได้: {e}")
                print("🎮 สลับไปใช้โหมดจำลองอัตโนมัติ")
                self.model = None
                self.use_simulation = True
        
        # กำหนดชื่อเหรียญบาทไทย (ภาษาไทย)
        self.coin_names_th = {
            '1baht': 'เหรียญ 1 บาท',
            '2baht': 'เหรียญ 2 บาท',
            '5baht': 'เหรียญ 5 บาท',
            '10baht': 'เหรียญ 10 บาท',
            'unknown': 'เหรียญ'
        }
        
        # ค่าเหรียญ
        self.coin_values = {
            '1baht': 1,
            '2baht': 2,
            '5baht': 5,
            '10baht': 10,
            'unknown': 0
        }
        
        # สีของแต่ละเหรียญ (BGR)
        self.coin_colors = {
            '1baht': (0, 255, 0),      # เขียว
            '2baht': (255, 128, 0),    # ส้ม
            '5baht': (0, 255, 255),    # เหลือง
            '10baht': (0, 0, 255),     # แดง
            'unknown': (255, 255, 255)  # ขาว
        }
        
        # เก็บสถิติ
        self.reset_stats()
        
    def _get_optimal_device(self):
        """เลือก device ที่เหมาะสม"""
        if torch.cuda.is_available():
            print("✅ พบ GPU, ใช้ CUDA")
            return torch.device('cuda')
        else:
            print("ℹ️ ไม่พบ GPU, ใช้ CPU")
            return torch.device('cpu')
    
    def reset_stats(self):
        """รีเซ็ตค่าสถิติ"""
        self.coin_counts = defaultdict(int)
        self.total_value = 0
        self.detection_history = []
        self.frame_count = 0
        self.total_frames = 0
        self.processing_times = []
    
    def _simulate_detection(self, frame):
        """
        จำลองการตรวจจับเหรียญ (สำหรับทดสอบ)
        สุ่มวางเหรียญปลอมในภาพ
        """
        h, w = frame.shape[:2]
        detections = []
        
        # สุ่มจำนวนเหรียญ (0-5 เหรียญ)
        num_coins = np.random.randint(0, 6)
        
        for i in range(num_coins):
            # สุ่มชนิดเหรียญ
            coin_type = np.random.choice(['1baht', '2baht', '5baht', '10baht'])
            
            # สุ่มตำแหน่ง
            x1 = np.random.randint(50, w-150)
            y1 = np.random.randint(50, h-150)
            x2 = x1 + np.random.randint(60, 100)
            y2 = y1 + np.random.randint(60, 100)
            
            # สุ่มความมั่นใจ
            confidence = np.random.uniform(0.7, 0.98)
            
            detections.append({
                'label': coin_type,
                'name_th': self.coin_names_th[coin_type],
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2),
                'value': self.coin_values[coin_type]
            })
        
        return detections
    
    def process_frame(self, frame):
        """
        ประมวลผลหนึ่งเฟรมภาพเพื่อตรวจจับเหรียญ
        """
        if frame is None or frame.size == 0:
            return None, None
            
        start_time = time.time()
        self.total_frames += 1
        
        frame_counts = defaultdict(int)
        frame_value = 0
        detections = []
        
        try:
            # ถ้าใช้โหมดจำลอง
            if self.use_simulation:
                detections = self._simulate_detection(frame)
            else:
                # รัน YOLO detection จริง
                results = self.model(frame, stream=True, verbose=False, imgsz=640)
                
                for result in results:
                    boxes = result.boxes
                    
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            confidence = float(box.conf[0])
                            class_id = int(box.cls[0])
                            
                            # ตรวจสอบว่าเป็นเหรียญหรือไม่ (class 44 ใน COCO)
                            if class_id == 44 and confidence > 0.5:
                                detections.append({
                                    'label': 'unknown',
                                    'name_th': self.coin_names_th['unknown'],
                                    'confidence': confidence,
                                    'bbox': (x1, y1, x2, y2),
                                    'value': 0
                                })
            
            # นับเหรียญและวาดลงบนภาพ
            for det in detections:
                label = det['label']
                name_th = det['name_th']
                value = det['value']
                confidence = det['confidence']
                x1, y1, x2, y2 = det['bbox']
                
                # นับเหรียญ
                frame_counts[label] += 1
                frame_value += value
                
                # สีของ Bounding Box
                color = self.coin_colors.get(label, (255, 255, 255))
                
                # วาด Bounding Box (เส้นหนา)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # สร้างข้อความภาษาไทย (ใช้ภาษาอังกฤษแทนเพราะ OpenCV ไม่รองรับภาษาไทยโดยตรง)
                if value > 0:
                    text = f"{value} Baht"
                else:
                    text = "Coin"
                
                # วาดพื้นหลังข้อความ
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                
                # วาดข้อความ
                cv2.putText(frame, text, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # แสดงความมั่นใจ
                conf_text = f"{confidence:.2f}"
                cv2.putText(frame, conf_text, (x1, y2 + 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # อัปเดตสถิติ
            for coin_type, count in frame_counts.items():
                if count > self.coin_counts[coin_type]:
                    self.coin_counts[coin_type] = count
            
            if frame_value > 0:
                self.total_value = max(self.total_value, frame_value)
            
            # เก็บประวัติ
            self.detection_history.append(frame_value)
            if len(self.detection_history) > 30:
                self.detection_history.pop(0)
            
            # คำนวณเวลาในการประมวลผล
            process_time = time.time() - start_time
            self.processing_times.append(process_time)
            if len(self.processing_times) > 30:
                self.processing_times.pop(0)
            
            avg_process_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
            
            detection_info = {
                'frame_counts': dict(frame_counts),
                'frame_value': frame_value,
                'avg_value': sum(self.detection_history) / len(self.detection_history) if self.detection_history else 0,
                'detections': detections,
                'process_time': process_time,
                'avg_process_time': avg_process_time,
                'total_frames': self.total_frames
            }
            
        except Exception as e:
            print(f"⚠️ Error processing frame: {e}")
            detection_info = {
                'frame_counts': {},
                'frame_value': 0,
                'avg_value': 0,
                'detections': [],
                'error': str(e)
            }
        
        return frame, detection_info
    
    def draw_info_panel(self, frame, detection_info, fps=None):
        """
        วาดแผงข้อมูลแสดงผลบนภาพ
        """
        if frame is None or frame.size == 0:
            return frame
            
        h, w = frame.shape[:2]
        
        # สร้างพื้นหลังโปร่งแสง
        overlay = frame.copy()
        panel_x, panel_y = 10, 10
        panel_w, panel_h = 380, 380
        
        # ปรับขนาด panel
        if h < 450:
            panel_h = 320
            
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_w, panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # วาดกรอบ
        cv2.rectangle(frame, (panel_x, panel_y), (panel_w, panel_h), (255, 215, 0), 2)
        
        # หัวข้อ
        title = "💰 ระบบตรวจจับเหรียญบาทไทย"
        cv2.putText(frame, title, (panel_x+20, panel_y+35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 215, 0), 2)
        
        y_pos = panel_y + 70
        
        # แสดงสถานะ
        status = "โหมดจำลอง" if self.use_simulation else "โหมดจริง"
        status_color = (0, 255, 255) if self.use_simulation else (0, 255, 0)
        cv2.putText(frame, f"สถานะ: {status}", (panel_x+20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        y_pos += 25
        
        # แสดง Device และ FPS
        device_text = f"Device: {str(self.device).upper()}"
        cv2.putText(frame, device_text, (panel_x+20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        y_pos += 20
        
        if fps:
            fps_text = f"FPS: {fps}"
            cv2.putText(frame, fps_text, (panel_x+20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
            y_pos += 20
        
        # แสดงเวลาในการประมวลผล
        process_time = detection_info.get('avg_process_time', 0)
        cv2.putText(frame, f"เวลา: {process_time*1000:.1f} ms", (panel_x+20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y_pos += 30
        
        # เส้นคั่น
        cv2.line(frame, (panel_x+15, y_pos), (panel_w-15, y_pos), (100, 100, 100), 1)
        y_pos += 15
        
        # แสดงจำนวนเหรียญ
        cv2.putText(frame, "📌 เหรียญที่พบ:", (panel_x+20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y_pos += 25
        
        frame_counts = detection_info.get('frame_counts', {})
        if frame_counts:
            for coin_type, count in frame_counts.items():
                if coin_type in self.coin_values:
                    value = self.coin_values[coin_type]
                    color = self.coin_colors.get(coin_type, (255, 255, 255))
                    
                    if value > 0:
                        text = f"  เหรียญ {value} บาท: {count} เหรียญ"
                    else:
                        text = f"  เหรียญไม่ทราบค่า: {count} เหรียญ"
                    
                    cv2.putText(frame, text, (panel_x+25, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_pos += 20
        else:
            cv2.putText(frame, "  ไม่พบเหรียญ", (panel_x+25, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            y_pos += 20
        
        # เส้นคั่น
        y_pos += 5
        cv2.line(frame, (panel_x+15, y_pos), (panel_w-15, y_pos), (100, 100, 100), 1)
        y_pos += 15
        
        # แสดงมูลค่ารวม
        frame_value = detection_info.get('frame_value', 0)
        avg_value = detection_info.get('avg_value', 0)
        
        if frame_value > 0:
            value_text = f"💰 มูลค่ารวม: {frame_value} บาท"
        else:
            value_text = "💰 มูลค่ารวม: 0 บาท"
            
        cv2.putText(frame, value_text, (panel_x+20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 215, 0), 2)
        y_pos += 25
        
        cv2.putText(frame, f"📊 ค่าเฉลี่ย: {avg_value:.1f} บาท", (panel_x+20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 20
        
        # แสดงสถิติรวม
        if self.total_value > 0:
            cv2.putText(frame, f"🏆 สูงสุด: {self.total_value} บาท", (panel_x+20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 100), 1)
        
        # เพิ่มคำแนะนำ
        help_y = panel_h - 40
        cv2.putText(frame, "กด: q=ออก, s=บันทึก, r=รีเซ็ต", (panel_x+20, help_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return frame

def test_stream_connection(url):
    """ทดสอบการเชื่อมต่อกับ MJPEG stream"""
    try:
        print(f"🔍 กำลังทดสอบการเชื่อมต่อ: {url}")
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0'},
            method='GET'
        )
        response = urllib.request.urlopen(req, timeout=5)
        content_type = response.headers.get('Content-Type', '')
        
        print(f"✅ ตอบสนอง: HTTP {response.status}")
        print(f"📋 Content-Type: {content_type}")
        
        # อ่านข้อมูลบางส่วนเพื่อทดสอบ
        initial_data = response.read(1024)
        if b'\xff\xd8' in initial_data:
            print("✅ พบ JPEG data ใน stream")
            return True
        else:
            print("⚠️ อาจไม่ใช่ MJPEG stream ที่ถูกต้อง")
            return True
            
    except Exception as e:
        print(f"❌ ไม่สามารถเชื่อมต่อ: {e}")
        return False

def main():
    """
    ฟังก์ชันหลัก
    """
    stream_url = "http://192.168.200.68:8080/video"
    
    print("=" * 60)
    print("💰 ระบบตรวจจับและนับเหรียญบาทไทย")
    print("=" * 60)
    
    # ทดสอบการเชื่อมต่อ
    if not test_stream_connection(stream_url):
        print("❌ ไม่สามารถเชื่อมต่อกับ stream ได้")
        return
    
    print("\n🚀 เริ่มการทำงาน...")
    
    # สร้าง MJPEG stream reader
    stream_reader = MJPEGStreamReader(stream_url)
    stream_reader.start()
    
    # รอให้ได้เฟรมแรก
    print("⏳ กำลังรอข้อมูลจากกล้อง...")
    timeout = 15
    start_time = time.time()
    first_frame = None
    
    while time.time() - start_time < timeout:
        frame = stream_reader.get_frame()
        if frame is not None and frame.size > 0:
            first_frame = frame
            print("✅ ได้รับเฟรมแรกแล้ว")
            break
        time.sleep(0.5)
        print(".", end="", flush=True)
    
    print()
    
    if first_frame is None:
        print("❌ ไม่ได้รับข้อมูลจากกล้อง")
        stream_reader.stop()
        return
    
    # เลือกโหมดการทำงาน
    print("\nเลือกโหมดการทำงาน:")
    print("1. โหมดจำลอง (สำหรับทดสอบ)")
    print("2. โหมดจริง (ต้องมีโมเดล)")
    
    mode = input("เลือก (1/2): ").strip()
    use_simulation = (mode != "2")
    
    # เริ่มต้นตัวตรวจจับเหรียญ
    try:
        detector = ThaiCoinDetector(model_path="", use_simulation=use_simulation)
    except Exception as e:
        print(f"❌ ไม่สามารถเริ่มระบบ: {e}")
        stream_reader.stop()
        return
    
    print("\n🎯 เริ่มการตรวจจับ...")
    print("   q: ออกจากโปรแกรม")
    print("   s: บันทึกภาพ")
    print("   r: รีเซ็ตสถิติ")
    print("-" * 60)
    
    last_frame_time = time.time()
    frame_display_interval = 1.0 / 30
    
    try:
        while True:
            current_time = time.time()
            
            # ดึงเฟรมล่าสุด
            frame = stream_reader.get_frame()
            
            if frame is not None and frame.size > 0:
                # ตรวจจับเหรียญ
                processed_frame, detection_info = detector.process_frame(frame.copy())
                
                if processed_frame is not None and processed_frame.size > 0:
                    # วาดข้อมูลแสดงผล
                    processed_frame = detector.draw_info_panel(
                        processed_frame, 
                        detection_info,
                        stream_reader.fps
                    )
                    
                    # แสดงผล
                    if current_time - last_frame_time >= frame_display_interval:
                        cv2.imshow('Thai Coin Detector', processed_frame)
                        last_frame_time = current_time
                    
                    # แสดงข้อมูลบน console ทุก 30 เฟรม
                    if detector.total_frames % 30 == 0:
                        frame_value = detection_info.get('frame_value', 0)
                        print(f"📊 เฟรมที่ {detector.total_frames}: พบ {len(detection_info.get('detections', []))} เหรียญ มูลค่า {frame_value} บาท")
            
            # รับค่าจากคีย์บอร์ด
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('q'):
                print("👋 หยุดการทำงาน")
                break
            elif key == ord('s'):
                filename = f"coin_detection_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                if 'processed_frame' in locals() and processed_frame is not None:
                    cv2.imwrite(filename, processed_frame)
                    print(f"💾 บันทึกภาพ: {filename}")
            elif key == ord('r'):
                detector.reset_stats()
                print("🔄 รีเซ็ตสถิติเรียบร้อย")
                
    except KeyboardInterrupt:
        print("\n👋 หยุดการทำงานโดยผู้ใช้")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
    finally:
        stream_reader.stop()
        cv2.destroyAllWindows()
        
        # แสดงสรุปผล
        print("\n" + "=" * 60)
        print("📊 สรุปผลการตรวจจับ")
        print("=" * 60)
        
        total_coins = 0
        total_value_sum = 0
        
        for coin_type, count in detector.coin_counts.items():
            if count > 0:
                value = detector.coin_values.get(coin_type, 0)
                if value > 0:
                    print(f"  เหรียญ {value} บาท: {count} เหรียญ")
                    total_coins += count
                    total_value_sum += value * count
                else:
                    print(f"  เหรียญไม่ทราบค่า: {count} เหรียญ")
                    total_coins += count
        
        print(f"  รวมทั้งสิ้น: {total_coins} เหรียญ")
        print(f"💰 มูลค่ารวมสูงสุดที่พบ: {detector.total_value} บาท")
        print(f"📊 ประมวลผลไปแล้ว: {detector.total_frames} เฟรม")
        print("=" * 60)

if __name__ == "__main__":
    main()