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
    
    def __init__(self, url, queue_size=5):  # ลด queue size เพื่อลด memory
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
                # เปิด connection พร้อม headers ที่จำเป็น
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
                        # อ่านข้อมูลจาก stream
                        chunk = self.stream.read(4096)
                        #print(chunk.status)  
                        
                        if not chunk:
                            print("⚠️ Stream สิ้นสุดลง")
                            break
                            
                        self.bytes_data += chunk
                        
                        # ค้นหา JPEG markers
                        start_marker = self.bytes_data.find(b'\xff\xd8')
                        end_marker = self.bytes_data.find(b'\xff\xd9')
                        
                        if start_marker != -1 and end_marker != -1 and end_marker > start_marker:
                            # ได้ JPEG ครบ 1 เฟรม
                            jpg_data = self.bytes_data[start_marker:end_marker+2]
                            self.bytes_data = self.bytes_data[end_marker+2:]
                            
                            try:
                                # แปลง JPEG เป็น numpy array
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
                                    
                                    # ใส่ frame ใน queue (ถ้า queue เต็ม จะแทนที่อันเก่า)
                                    if self.frame_queue.full():
                                        try:
                                            self.frame_queue.get_nowait()
                                        except:
                                            pass
                                    self.frame_queue.put(frame)
                                    
                            except Exception as e:
                                # ถ้า decode ไม่ได้ ให้ข้ามเฟรมนี้ไป
                                continue
                                
                    except Exception as e:
                        print(f"⚠️ Error reading stream: {e}")
                        break
                        
            except Exception as e:
                self.connection_attempts += 1
                print(f"⚠️ ไม่สามารถเชื่อมต่อ MJPEG stream (ครั้งที่ {self.connection_attempts}): {e}")
                
                if self.connection_attempts >= self.max_attempts:
                    print("❌ ไม่สามารถเชื่อมต่อได้หลังจากพยายามหลายครั้ง")
                    self.running = False
                    break
                    
                # รอก่อนลองใหม่
                time.sleep(2)
            
            # ถ้า connection หลุด ให้ลองใหม่
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
    def __init__(self, model_path=None):
        """
        กำหนดค่าเริ่มต้นสำหรับการตรวจจับเหรียญบาทไทย
        
        Args:
            model_path: พาธไปยังโมเดลที่เทรนมาเฉพาะ (ถ้าไม่มีจะใช้โมเดลพื้นฐาน)
        """
        # ตรวจสอบ device และตั้งค่าที่เหมาะสม
        self.device = self._get_optimal_device()
        print(f"💻 ใช้ device: {self.device}")
        
        # เลือกใช้โมเดล YOLO
        if model_path and model_path != "":
            print(f"⚙️ กำลังโหลดโมเดล: {model_path}")
            self.model = YOLO(model_path)
            #print(self.model.names)
        else:
            print("⚙️ กำลังโหลดโมเดลพื้นฐาน YOLOv8n...")
            self.model = YOLO('yolov8n.pt')
            print("⚠️ กำลังใช้โมเดลพื้นฐานจาก COCO ซึ่งอาจตรวจจับเหรียญได้ไม่ดีเท่าที่ควร")
            print("   แนะนำให้เทรนโมเดลเฉพาะสำหรับเหรียญบาทไทย")
        
        # ย้ายโมเดลไปยัง device ที่เหมาะสม และใช้ float32 เสมอ
        self.model.to(self.device)
        self.model.model.half = False  # ปิด half precision
        
        # กำหนดชื่อคลาสสำหรับเหรียญบาทไทย
        self.coin_classes = {
            0: '1 Baht',
            1: '2 Baht',
            2: '5 Baht',
            3: '10 Baht'
        }
        
        # ค่าเหรียญ (บาท)
        self.coin_values = {
            '1 Baht': 1,
            '2 Baht': 2,
            '5 Baht': 5,
            '10 Baht': 10,
            'Coin': 1  # สำหรับกรณีที่ไม่รู้ชนิด
        }
        
        # เก็บสถิติ
        self.reset_stats()


        
    def _get_optimal_device(self):
        """เลือก device ที่เหมาะสมสำหรับการรันโมเดล"""
        if torch.cuda.is_available():
            # ถ้ามี GPU ให้ใช้ GPU
            device = torch.device('cuda')
            print("✅ พบ GPU, ใช้ CUDA")
            # ตรวจสอบว่า GPU รองรับ half precision หรือไม่
            try:
                test_tensor = torch.ones(1, 3, 640, 640, device=device)
                test_tensor.half()
                print("   GPU รองรับ half precision")
            except:
                print("   GPU ไม่รองรับ half precision, จะใช้ float32")
            return device
        else:
            # ถ้าไม่มี GPU ให้ใช้ CPU
            print("ℹ️ ไม่พบ GPU, ใช้ CPU (อาจช้าหน่อย)")
            return torch.device('cpu')
    
    def reset_stats(self):
        """รีเซ็ตค่าสถิติ"""
        self.coin_counts = defaultdict(int)
        self.total_value = 0
        self.detection_history = []
        self.frame_count = 0
        self.total_frames = 0
        self.processing_times = []
        
    def process_frame(self, frame):
        """
        ประมวลผลหนึ่งเฟรมภาพเพื่อตรวจจับเหรียญ
        
        Args:
            frame: ภาพจากกล้อง (numpy array)
            
        Returns:
            frame_with_detections: ภาพที่วาด Bounding Box และข้อความแล้ว
            detection_info: ข้อมูลการตรวจจับ (dict)
        """
        if frame is None or frame.size == 0:
            return None, None
            
        start_time = time.time()
        self.total_frames += 1
        
        # รีเซ็ตค่า counts ก่อนนับใหม่ในเฟรมนี้
        frame_counts = defaultdict(int)
        frame_value = 0
        detections = []
        
        try:
            # รัน YOLO detection โดยไม่ใช้ half precision
            #results = self.model.predict(frame, stream=False, verbose=False, imgsz=640)
            results = self.model.track(frame, persist= True , stream=False, verbose=False, imgsz=640)
            
            #print(f"🔍 ตรวจจับเสร็จใน {time.time() - start_time:.2f} วินาที, พบ {len(results)} ผลลัพธ์")
            
            #print(f"📋 names: {results[0].boxes.cls}")  
            # วาดผลลัพธ์บนภาพ
            for result in results:
                boxes = result.boxes
                
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # ดึงพิกัด Bounding Box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # ดึง confidence และ class id
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # ตรวจสอบว่าเป็นเหรียญหรือไม่
                        label = None
                        value = 0
                        
                        # กรณีใช้โมเดล COCO (class 44 คือ coin)
                        if class_id == 44:
                            label = "Coin"
                            value = 1
                        # กรณีใช้โมเดลที่เทรนเฉพาะ
                        elif class_id in self.coin_classes:
                            label = self.coin_classes[class_id]
                            value = self.coin_values.get(label, 0)
                        
                        # ถ้าเจอเหรียญและ confidence สูงพอ
                        if label and confidence > 0.5:
                            # นับเหรียญ
                            frame_counts[label] += 1
                            frame_value += value
                            
                            # เก็บข้อมูล detection
                            detections.append({
                                'label': label,
                                'confidence': confidence,
                                'bbox': (x1, y1, x2, y2),
                                'value': value
                            })
                            
                            # สีของ Bounding Box
                            color_map = {
                                '1 Baht': (0, 255, 0),    # เขียว
                                '2 Baht': (255, 128, 0),  # ส้ม
                                '5 Baht': (0, 255, 255),  # เหลือง
                                '10 Baht': (0, 0, 255),   # แดง
                                'Coin': (255, 255, 0)      # ฟ้า
                            }
                            color = color_map.get(label, (255, 255, 255))
                            
                            # วาด Bounding Box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # สร้างข้อความ
                            text = f"{label} ({value})"
                            
                            # วาดพื้นหลังข้อความ
                            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                            
                            # วาดข้อความ
                            cv2.putText(frame, text, (x1, y1 - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                            
                            # แสดง confidence
                            conf_text = f"{confidence:.2f}"
                            cv2.putText(frame, conf_text, (x1, y2 + 20),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # อัปเดตสถิติ
            for coin_type, count in frame_counts.items():
                if count > self.coin_counts[coin_type]:
                    self.coin_counts[coin_type] = count
            
            if frame_value > 0:
                self.total_value = max(self.total_value, frame_value)
            
            # เก็บประวัติ
            self.detection_history.append(frame_value)
            if len(self.detection_history) > 30:  # เก็บ 30 เฟรมล่าสุด
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
        
        Args:
            frame: ภาพต้นฉบับ
            detection_info: ข้อมูลการตรวจจับ
            fps: ค่า FPS (ถ้ามี)
        """
        if frame is None or frame.size == 0:
            return frame
            
        h, w = frame.shape[:2]
        
        # สร้างพื้นหลังโปร่งแสง (ด้านซ้ายบน)
        overlay = frame.copy()
        panel_x, panel_y = 10, 10
        panel_w, panel_h = 350, 350
        
        # ปรับขนาด panel ถ้าหน้าจอเล็ก
        if h < 400:
            panel_h = 300
            
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_w, panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # วาดกรอบ
        cv2.rectangle(frame, (panel_x, panel_y), (panel_w, panel_h), (100, 100, 100), 1)
        
        # หัวข้อ
        cv2.putText(frame, " Thai Coin Detector", (panel_x+15, panel_y+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 215, 0), 2)
        
        y_pos = panel_y + 50
        
        # แสดง Device และ FPS
        device_text = f" Device: {str(self.device).upper()}"
        cv2.putText(frame, device_text, (panel_x+15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y_pos += 20
        
        if fps:
            fps_text = f" FPS: {fps}"
            cv2.putText(frame, fps_text, (panel_x+15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            y_pos += 20
        
        # แสดงเวลาในการประมวลผล
        process_time = detection_info.get('avg_process_time', 0)
        cv2.putText(frame, f" Process: {process_time*1000:.1f}ms", (panel_x+15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 25
        
        # เส้นคั่น
        cv2.line(frame, (panel_x+10, y_pos), (panel_w-10, y_pos), (100, 100, 100), 1)
        y_pos += 15
        
        # แสดงจำนวนเหรียญในเฟรมปัจจุบัน
        cv2.putText(frame, " Current Frame:", (panel_x+15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20
        
        frame_counts = detection_info.get('frame_counts', {})
        if frame_counts:
            for coin_type, count in frame_counts.items():
                value = self.coin_values.get(coin_type, 1)
                color_map = {
                    '1 Baht': (0, 255, 0),
                    '2 Baht': (255, 128, 0),
                    '5 Baht': (0, 255, 255),
                    '10 Baht': (0, 0, 255),
                    'Coin': (255, 255, 0)
                }
                color = color_map.get(coin_type, (255, 255, 255))
                
                text = f"  {coin_type}: {count} (={value*count} Baht)"
                cv2.putText(frame, text, (panel_x+20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_pos += 20
        else:
            cv2.putText(frame, "  No coins", (panel_x+20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            y_pos += 20
        
        # เส้นคั่น
        y_pos += 5
        cv2.line(frame, (panel_x+10, y_pos), (panel_w-10, y_pos), (100, 100, 100), 1)
        y_pos += 15
        
        # แสดงมูลค่ารวม
        frame_value = detection_info.get('frame_value', 0)
        avg_value = detection_info.get('avg_value', 0)
        
        cv2.putText(frame, f" Current: {frame_value} Baht", (panel_x+15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 215, 0), 2)
        y_pos += 25
        
        cv2.putText(frame, f" Average: {avg_value:.1f} Baht", (panel_x+15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 20
        
        # แสดงสถิติรวม
        if self.total_value > 0:
            cv2.putText(frame, f" Max: {self.total_value} Baht", (panel_x+15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 100), 1)
        
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
        if b'\xff\xd8' in initial_data:  # มี JPEG marker
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
    ฟังก์ชันหลักสำหรับเชื่อมต่อ MJPEG stream และเริ่มการตรวจจับ
    """
    # URL ของกล้อง MJPEG
    stream_url = "http://192.168.200.221:4747/video"
    
    print("=" * 60)
    print("💰 Thai Coin Detector with MJPEG Stream")
    print("=" * 60)
    
    # ตรวจสอบ PyTorch version
    print(f"📦 PyTorch version: {torch.__version__}")
    print(f"📦 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"📦 CUDA version: {torch.version.cuda}")
        print(f"📦 GPU: {torch.cuda.get_device_name(0)}")
    
    # ทดสอบการเชื่อมต่อ
    if not test_stream_connection(stream_url):
        print("❌ ไม่สามารถเชื่อมต่อกับ stream ได้")
        print("   ตรวจสอบ: ")
        print("   - IP Address ถูกต้อง (192.168.200.68)")
        print("   - พอร์ต 8080 ถูกต้อง")
        print("   - กล้องเปิดอยู่และกำลังส่ง MJPEG stream")
        print("   - ไม่มี Firewall ปิดกั้น")
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
        print("❌ ไม่ได้รับข้อมูลจากกล้องภายในเวลาที่กำหนด")
        stream_reader.stop()
        return
    
    # เริ่มต้นตัวตรวจจับเหรียญ
    try:
        detector = ThaiCoinDetector(model_path="best.pt")  # เปลี่ยนเป็น thai_coin_model.pt ถ้ามี
    except Exception as e:
        print(f"❌ ไม่สามารถโหลดโมเดล: {e}")
        stream_reader.stop()
        return
    
    print("\n🎯 เริ่มการตรวจจับ กด:")
    print("   'q' - ออกจากโปรแกรม")
    print("   's' - บันทึกภาพปัจจุบัน")
    print("   'r' - รีเซ็ตสถิติ")
    print("-" * 60)
    
    last_frame_time = time.time()
    frame_display_interval = 1.0 / 30  # จำกัดการแสดงผลที่ 30 FPS
    
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
                    
                    # แสดงผล (จำกัด FPS การแสดงผล)
                    if current_time - last_frame_time >= frame_display_interval:
                        cv2.imshow('Thai Coin Detector - MJPEG Stream', processed_frame)
                        last_frame_time = current_time
                    
                    # เก็บเฟรมล่าสุด
                    last_frame = frame
            
            # รับค่าจากคีย์บอร์ด (ลดความถี่เพื่อลด CPU usage)
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('q'):
                print("👋 หยุดการทำงาน")
                break
            elif key == ord('s'):
                # บันทึกภาพ
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
        # ทำความสะอาด
        stream_reader.stop()
        cv2.destroyAllWindows()
        
        # แสดงสรุปผล
        print("\n" + "=" * 60)
        print("📊 สรุปผลการตรวจจับ:")
        for coin_type, count in detector.coin_counts.items():
            if count > 0:
                value = detector.coin_values.get(coin_type, 1)
                print(f"  {coin_type}: {count} เหรียญ (มูลค่า {value*count} บาท)")
        print(f"💰 มูลค่ารวมสูงสุดที่พบ: {detector.total_value} บาท")
        print(f"📊 ประมวลผลไปแล้ว: {detector.total_frames} เฟรม")
        print("=" * 60)

if __name__ == "__main__":
    main()