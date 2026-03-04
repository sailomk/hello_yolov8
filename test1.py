import cv2
import socket
import sys

print("=" * 50)
print("DIAGNOSING CONNECTION ISSUE")
print("=" * 50)

# Test 1: Can Python establish a raw socket connection?
print("\n[Test 1] Testing raw socket connection to 127.0.0.1:8081...")
try:
    test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    test_sock.settimeout(3)
    result = test_sock.connect_ex(('127.0.0.1', 8081))  # Returns 0 on success
    
    if result == 0:
        print("✅ Raw socket connection SUCCESSFUL!")
        print("   Your Python CAN reach socat")
        
        # Try to receive any data
        try:
            data = test_sock.recv(1024)
            print(f"   Received {len(data)} bytes from server")
        except socket.timeout:
            print("   No data received (timeout) - server didn't send anything")
    else:
        print(f"❌ Raw socket connection FAILED with error code: {result}")
        print("   Your Python CANNOT reach socat")
        
        # Try to diagnose why
        if result == 61:
            print("   Error 61: Connection refused - socat might not be running")
        elif result == 8:
            print("   Error 8: Address family not supported - try 127.0.0.1 instead of localhost")
    test_sock.close()
    
except Exception as e:
    print(f"❌ Socket test crashed: {e}")

# Test 2: Try different OpenCV backends
print("\n[Test 2] Testing OpenCV VideoCapture with different backends...")

urls_to_try = [
    "http://127.0.0.1:8081/video",
    "http://127.0.0.1:8081/",
    "http://127.0.0.1:8081",
    "http://localhost:8081/video",
]

backends = [
    (cv2.CAP_ANY, "CAP_ANY (default)"),
    # Uncomment if you have these backends:
    # (cv2.CAP_FFMPEG, "CAP_FFMPEG"),
    # (cv2.CAP_GSTREAMER, "CAP_GSTREAMER"),
]

for url in urls_to_try:
    for backend, backend_name in backends:
        print(f"\nTrying: {url} with {backend_name}")
        
        # Create VideoCapture with explicit backend
        cap = cv2.VideoCapture(url, backend)
        
        # Check if opened
        if cap.isOpened():
            print(f"✅ VideoCapture OPENED successfully!")
            
            # Try to read a frame
            print("   Attempting to read first frame...")
            ret, frame = cap.read()
            
            if ret and frame is not None:
                print(f"   ✅ Frame captured! Shape: {frame.shape}")
                
                # Show the frame briefly
                cv2.imshow('Test Frame', frame)
                print("   Press any key on the image window to continue...")
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
                
                # Success! Use this URL and backend
                print(f"\n🎯 SUCCESS! Use: url='{url}', backend={backend_name}")
                cap.release()
                sys.exit(0)
            else:
                print(f"   ❌ Failed to read frame (ret={ret}, frame is None)")
        else:
            print(f"   ❌ VideoCapture failed to open")
        
        cap.release()

print("\n" + "=" * 50)
print("❌ ALL TESTS FAILED")
print("=" * 50)
print("\nPossible issues:")
print("1. socat might not be running - check with: ps aux | grep socat")
print("2. The server at 192.168.200.179:8080 might not be sending video data")
print("3. The video format might not be compatible with OpenCV")
print("4. OpenCV might be compiled without necessary codecs")