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
                            print("⚠️ Stream สิ้นสุดลง")
                            break
                            
                        self.bytes_data += chunk
                        
                        start_marker = self.bytes_data.find(b'\xff\xd8')
                        end_marker = self.bytes_data.find(b'\xff\xd9')
                        
                        if start_marker != -1 and end_marker != -1 and end_marker > start_marker:
                            jpg_data = self.bytes_data[start_marker:end_marker+2]
                            self.bytes_data = self.bytes_data[end_marker+2:]
                            
                            try:
                                frame_array = np.frombuffer(jpg_data, dtype=np.uint8)
                                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                                
                                if frame is not None and frame.size > 0:
                                    self.frame_count += 1
                                    current_time = time.time()
                                    if current_time - self.last_time >= 1.0:
                                        self.fps = self.frame_count
                                        self.frame_count = 0
                                        self.last_time = current_time
                                    
                                    if self.frame_queue.full():
                                        try:
                                            self.frame_queue.get_nowait()
                                        except:
                                            pass
                                    self.frame_queue.put(frame)
                                    
                            except Exception as e:
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
    def __init__(self, model_path=None):
        """
        กำหนดค่าเริ่มต้นสำหรับการตรวจจับเหรียญบาทไทย
        
        Args:
            model_path: พาธไปยังโมเดลที่เทรนมาเฉพาะ
        """
        # ตรวจสอบ device
        self.device = self._get_optimal_device()
        print(f"💻 ใช้ device: {self.device}")
        
        # โหลดโมเดล
        if model_path and model_path != "":
            print(f"⚙️ กำลังโหลดโมเดล: {model_path}")
            self.model = YOLO(model_path)
        else:
            print("⚙️ กำลังโหลดโมเดลพื้นฐาน YOLOv8n...")
            self.model = YOLO('yolov8n.pt')
            print("⚠️ กำลังใช้โมเดลพื้นฐานจาก COCO")
        
        self.model.to(self.device)
        self.model.model.half = False
        
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
            'Coin': 1
        }
        
        # ขนาดเส้นผ่านศูนย์กลางมาตรฐานของเหรียญบาทไทย (มม.)
        self.coin_diameters = {
            '1 Baht': 20.0,
            '2 Baht': 21.75,
            '5 Baht': 24.0,
            '10 Baht': 25.0,  # ค่าเฉลี่ยระหว่างแบบเก่า(26)และใหม่(24)
            'Coin': 22.0
        }
        
        # ค่า tolerance สำหรับขนาด (±%)
        self.size_tolerance = 0.15  # 15%
        
        # ตัวแปรสำหรับ calibration
        self.pixels_per_mm = None
        self.calibrated = False
        self.use_size_verification = True
        self.calibration_mode = False
        
        # เก็บสถิติ
        self.reset_stats()
        
        print("\n📏 ระบบตรวจสอบขนาดเหรียญ:")
        print("   - กด 'c' เพื่อ calibrate ด้วยเหรียญ 1 บาท")
        print("   - กด '1-4' เพื่อ calibrate ด้วยเหรียญที่เลือก")
        print("   - กด 't' เพื่อเปิด/ปิดการตรวจสอบด้วยขนาด")
        print("-" * 60)
        
    def _get_optimal_device(self):
        """เลือก device ที่เหมาะสม"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("✅ พบ GPU, ใช้ CUDA")
            return device
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
    
    def calibrate_with_known_coin(self, frame, bbox, known_type='1 Baht'):
        """
        calibrate ระบบโดยใช้เหรียญที่รู้ชนิด
        
        Args:
            frame: ภาพจากกล้อง
            bbox: bounding box ของเหรียญ (x1, y1, x2, y2)
            known_type: ชนิดเหรียญที่รู้
        """
        x1, y1, x2, y2 = bbox
        pixel_diameter = max(x2 - x1, y2 - y1)
        
        # คำนวณอัตราส่วน pixels ต่อ มม.
        actual_diameter = self.coin_diameters[known_type]
        self.pixels_per_mm = pixel_diameter / actual_diameter
        
        self.calibrated = True
        print(f"\n✅ Calibrated สำเร็จ!")
        print(f"📏 ขนาดในภาพ: {pixel_diameter:.1f} pixels")
        print(f"📏 ขนาดจริง: {actual_diameter} mm")
        print(f"📐 อัตราส่วน: {self.pixels_per_mm:.2f} pixels/mm")
        
        # วาดวงกลมแสดงขนาดที่ calibrate
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        radius = int(pixel_diameter / 2)
        cv2.circle(frame, (center_x, center_y), radius, (0, 255, 255), 2)
        cv2.putText(frame, "CALIBRATED", (center_x - 50, center_y - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return self.pixels_per_mm
    
    def verify_coin_by_size(self, bbox, predicted_label, confidence):
        """
        ตรวจสอบความถูกต้องของเหรียญโดยใช้ขนาด
        
        Args:
            bbox: bounding box (x1, y1, x2, y2)
            predicted_label: ชนิดที่โมเดลทำนาย
            confidence: ค่าความเชื่อมั่นจากโมเดล
            
        Returns:
            adjusted_label: ชนิดที่ปรับปรุงแล้ว
            adjusted_confidence: ความเชื่อมั่นที่ปรับแล้ว
            size_ratio: อัตราส่วนขนาด
            estimated_mm: ขนาดโดยประมาณ (มม.)
        """
        if not self.calibrated or not self.use_size_verification:
            return predicted_label, confidence, 1.0, 0
            
        x1, y1, x2, y2 = bbox
        pixel_diameter = max(x2 - x1, y2 - y1)
        
        # คำนวณขนาดจริงโดยประมาณ (มม.)
        estimated_diameter = pixel_diameter / self.pixels_per_mm
        
        # หาเหรียญที่มีขนาดใกล้เคียงที่สุด
        best_match = predicted_label
        best_match_score = 0
        best_ratio = 1.0
        
        for coin_type, standard_diameter in self.coin_diameters.items():
            # คำนวณอัตราส่วน
            ratio = estimated_diameter / standard_diameter
            
            # ถ้าอยู่ใน tolerance ที่กำหนด
            if abs(1 - ratio) <= self.size_tolerance:
                # คะแนนความใกล้เคียง (1 = ตรงเป๊ะ)
                match_score = 1 - abs(1 - ratio)
                
                # รวมกับ confidence เดิม
                if coin_type == predicted_label:
                    match_score = (match_score + confidence) / 2
                else:
                    # ลดคะแนนถ้าคนละชนิด
                    match_score = match_score * 0.8
                
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_match = coin_type
                    best_ratio = ratio
        
        # ถ้าไม่มีเหรียญไหนอยู่ใน tolerance
        if best_match_score == 0:
            # หาอันที่ใกล้เคียงที่สุด
            closest_type = min(self.coin_diameters.keys(), 
                             key=lambda x: abs(estimated_diameter - self.coin_diameters[x]))
            best_match = closest_type
            best_match_score = confidence * 0.5
            best_ratio = estimated_diameter / self.coin_diameters[closest_type]
        
        return best_match, min(best_match_score, 1.0), best_ratio, estimated_diameter
    
    def process_frame(self, frame):
        """
        ประมวลผลหนึ่งเฟรมภาพเพื่อตรวจจับเหรียญ
        
        Args:
            frame: ภาพจากกล้อง
            
        Returns:
            frame_with_detections: ภาพที่วาดแล้ว
            detection_info: ข้อมูลการตรวจจับ
        """
        if frame is None or frame.size == 0:
            return None, None
            
        start_time = time.time()
        self.total_frames += 1
        
        frame_counts = defaultdict(int)
        frame_value = 0
        detections = []
        
        try:
            # รัน YOLO detection
            results = self.model.track(frame, persist=True, stream=False, verbose=False, imgsz=640)
            
            for result in results:
                boxes = result.boxes
                
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # ตรวจสอบว่าเป็นเหรียญ
                        label = None
                        value = 0
                        
                        if class_id == 44:  # COCO class for coin
                            label = "Coin"
                            value = 1
                        elif class_id in self.coin_classes:
                            label = self.coin_classes[class_id]
                            value = self.coin_values.get(label, 0)
                        
                        if label and confidence > 0.5:
                            # ตรวจสอบด้วยขนาด (ถ้า calibrated แล้ว)
                            if self.calibrated and self.use_size_verification:
                                adjusted_label, adjusted_confidence, size_ratio, estimated_mm = self.verify_coin_by_size(
                                    (x1, y1, x2, y2), label, confidence
                                )
                                
                                # ปรับ label ตามขนาด
                                if adjusted_label != label:
                                    print(f"🔍 ปรับ label: {label} -> {adjusted_label} (ratio: {size_ratio:.2f})")
                                    label = adjusted_label
                                    confidence = adjusted_confidence
                                    value = self.coin_values.get(label, 1)
                                
                                # เก็บขนาดที่คำนวณได้
                                coin_size = estimated_mm
                            else:
                                # คำนวณขนาดคร่าวๆ (pixels)
                                pixel_diameter = max(x2 - x1, y2 - y1)
                                coin_size = pixel_diameter
                            
                            # นับเหรียญ
                            frame_counts[label] += 1
                            frame_value += value
                            
                            detections.append({
                                'label': label,
                                'confidence': confidence,
                                'bbox': (x1, y1, x2, y2),
                                'value': value,
                                'size': coin_size,
                                'size_unit': 'mm' if self.calibrated else 'pixels'
                            })
                            
                            # สีของ Bounding Box
                            color_map = {
                                '1 Baht': (0, 255, 0),
                                '2 Baht': (255, 128, 0),
                                '5 Baht': (0, 255, 255),
                                '10 Baht': (0, 0, 255),
                                'Coin': (255, 255, 0)
                            }
                            color = color_map.get(label, (255, 255, 255))
                            
                            # วาด Bounding Box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # สร้างข้อความหลัก
                            text = f"{label} ({value} B)"
                            
                            # วาดพื้นหลังข้อความหลัก
                            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                            
                            # วาดข้อความหลัก
                            cv2.putText(frame, text, (x1, y1 - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                            
                            # แสดง confidence
                            conf_text = f"{confidence:.2f}"
                            cv2.putText(frame, conf_text, (x1, y2 + 15),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                            
                            # แสดงขนาด
                            if self.calibrated:
                                size_text = f"{coin_size:.1f}mm"
                                cv2.putText(frame, size_text, (x1, y2 + 30),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                            
                            # วาดจุดศูนย์กลาง
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            cv2.circle(frame, (center_x, center_y), 3, (0, 255, 255), -1)
            
            # อัปเดตสถิติ
            for coin_type, count in frame_counts.items():
                if count > self.coin_counts[coin_type]:
                    self.coin_counts[coin_type] = count
            
            if frame_value > 0:
                self.total_value = max(self.total_value, frame_value)
            
            self.detection_history.append(frame_value)
            if len(self.detection_history) > 30:
                self.detection_history.pop(0)
            
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
                'total_frames': self.total_frames,
                'calibrated': self.calibrated,
                'use_size_verification': self.use_size_verification
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
        panel_w, panel_h = 380, 420
        
        if h < 500:
            panel_h = 400
            
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_w, panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        cv2.rectangle(frame, (panel_x, panel_y), (panel_w, panel_h), (100, 100, 100), 1)
        
        # หัวข้อ
        cv2.putText(frame, " Thai Coin Detector", (panel_x+15, panel_y+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 215, 0), 2)
        
        y_pos = panel_y + 50
        
        # Device และ FPS
        device_text = f" Device: {str(self.device).upper()}"
        cv2.putText(frame, device_text, (panel_x+15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y_pos += 20
        
        if fps:
            fps_text = f" Stream FPS: {fps}"
            cv2.putText(frame, fps_text, (panel_x+15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            y_pos += 20
        
        # เวลาประมวลผล
        process_time = detection_info.get('avg_process_time', 0)
        cv2.putText(frame, f" Process: {process_time*1000:.1f}ms", (panel_x+15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 25
        
        # สถานะ calibration
        cal_status = "✅ Calibrated" if self.calibrated else "❌ Not Calibrated"
        cal_color = (0, 255, 0) if self.calibrated else (0, 0, 255)
        cv2.putText(frame, f" {cal_status}", (panel_x+15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, cal_color, 1)
        y_pos += 20
        
        if self.calibrated:
            cv2.putText(frame, f" Scale: {self.pixels_per_mm:.2f} px/mm", (panel_x+15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_pos += 20
            
            size_status = "✅ Size Check ON" if self.use_size_verification else "❌ Size Check OFF"
            size_color = (0, 255, 0) if self.use_size_verification else (0, 0, 255)
            cv2.putText(frame, f" {size_status}", (panel_x+15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, size_color, 1)
            y_pos += 20
        
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
                
                text = f"  {coin_type}: {count} (={value*count} B)"
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
        
        # มูลค่ารวม
        frame_value = detection_info.get('frame_value', 0)
        avg_value = detection_info.get('avg_value', 0)
        
        cv2.putText(frame, f" Current: {frame_value} Baht", (panel_x+15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 215, 0), 2)
        y_pos += 25
        
        cv2.putText(frame, f" Average: {avg_value:.1f} Baht", (panel_x+15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 20
        
        if self.total_value > 0:
            cv2.putText(frame, f" Max: {self.total_value} Baht", (panel_x+15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 100), 1)
        
        # คำแนะนำการใช้งาน
        help_y = panel_h - 60
        cv2.putText(frame, " Commands:", (panel_x+15, help_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(frame, "  q: Quit  |  c: Calibrate", (panel_x+15, help_y+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, "  1-4: Cal with coin  |  t: Toggle size check", (panel_x+15, help_y+30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
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
    ฟังก์ชันหลักสำหรับเชื่อมต่อ MJPEG stream และเริ่มการตรวจจับ
    """
    # URL ของกล้อง MJPEG
    stream_url = "http://192.168.200.221:4747/video"
    
    print("=" * 60)
    print("💰 Thai Coin Detector with Size Verification")
    print("=" * 60)
    
    # ตรวจสอบ PyTorch
    print(f"📦 PyTorch version: {torch.__version__}")
    print(f"📦 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"📦 GPU: {torch.cuda.get_device_name(0)}")
    
    # ทดสอบการเชื่อมต่อ
    if not test_stream_connection(stream_url):
        print("❌ ไม่สามารถเชื่อมต่อกับ stream ได้")
        print("   ตรวจสอบ: ")
        print("   - IP Address ถูกต้อง")
        print("   - พอร์ตถูกต้อง")
        print("   - กล้องเปิดอยู่และกำลังส่ง MJPEG stream")
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
        detector = ThaiCoinDetector(model_path="best.pt")  # เปลี่ยนเป็นพาธโมเดลของคุณ
    except Exception as e:
        print(f"❌ ไม่สามารถโหลดโมเดล: {e}")
        stream_reader.stop()
        return
    
    print("\n🎯 เริ่มการตรวจจับ กด:")
    print("   'q' - ออกจากโปรแกรม")
    print("   's' - บันทึกภาพปัจจุบัน")
    print("   'r' - รีเซ็ตสถิติ")
    print("   'c' - calibrate ด้วยเหรียญ 1 บาท")
    print("   '1-4' - calibrate ด้วยเหรียญที่เลือก (1=1บ,2=2บ,3=5บ,4=10บ)")
    print("   't' - เปิด/ปิดการตรวจสอบด้วยขนาด")
    print("-" * 60)
    
    last_frame_time = time.time()
    frame_display_interval = 1.0 / 30
    
    try:
        while True:
            current_time = time.time()
            
            frame = stream_reader.get_frame()
            
            if frame is not None and frame.size > 0:
                processed_frame, detection_info = detector.process_frame(frame.copy())
                
                if processed_frame is not None and processed_frame.size > 0:
                    processed_frame = detector.draw_info_panel(
                        processed_frame, 
                        detection_info,
                        stream_reader.fps
                    )
                    
                    if current_time - last_frame_time >= frame_display_interval:
                        cv2.imshow('Thai Coin Detector - MJPEG Stream', processed_frame)
                        last_frame_time = current_time
                    
                    last_frame = frame
            
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
            elif key == ord('c') and detection_info and detection_info.get('detections'):
                # calibrate ด้วยเหรียญแรกที่เจอ (ใช้เหรียญ 1 บาท)
                first_detection = detection_info['detections'][0]
                detector.calibrate_with_known_coin(frame, first_detection['bbox'], '1 Baht')
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')] and detection_info and detection_info.get('detections'):
                coin_types = ['1 Baht', '2 Baht', '5 Baht', '10 Baht']
                idx = key - ord('1')
                if idx < len(coin_types):
                    first_detection = detection_info['detections'][0]
                    detector.calibrate_with_known_coin(frame, first_detection['bbox'], coin_types[idx])
            elif key == ord('t'):
                detector.use_size_verification = not detector.use_size_verification
                status = "เปิด" if detector.use_size_verification else "ปิด"
                print(f"🔍 การตรวจสอบด้วยขนาด: {status}")
                
    except KeyboardInterrupt:
        print("\n👋 หยุดการทำงานโดยผู้ใช้")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
    finally:
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