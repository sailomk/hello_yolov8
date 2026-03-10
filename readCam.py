import cv2
import numpy as np
import urllib.request
import io
from PIL import Image
import threading
from queue import Queue

class MJPEGReader:
    def __init__(self, url):
        self.url = url
        self.frame_queue = Queue(maxsize=10)
        self.running = True
        self.thread = threading.Thread(target=self._stream_reader, daemon=True)
        self.thread.start()
    
    def _stream_reader(self):
        stream = urllib.request.urlopen(self.url)
        bytes_data = b''
        
        while self.running:
            try:
                bytes_data += stream.read(1024)
                a = bytes_data.find(b'\xff\xd8')  # start of JPEG
                b = bytes_data.find(b'\xff\xd9')  # end of JPEG
                
                if a != -1 and b != -1 and b > a:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    
                    # แปลง JPEG → OpenCV image
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    
                    if frame is not None and not self.frame_queue.full():
                        self.frame_queue.put(frame)
                        
            except Exception as e:
                print(f"Error: {e}")
                break
    
    def read(self):
        if not self.frame_queue.empty():
            return True, self.frame_queue.get()
        return False, None

# ใช้งาน
reader = MJPEGReader("http://192.168.118.186:4747/video")

while True:
    ret, frame = reader.read()
    if ret:
        print(len(frame))  # แสดงขนาดของเฟรม
        cv2.imshow('Manual MJPEG Reader', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        reader.running = False
        break

cv2.destroyAllWindows()