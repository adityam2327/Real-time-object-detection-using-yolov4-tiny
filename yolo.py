import cv2
import numpy as np
import urllib.request
import os
import time
import requests
import threading
import socket

def download_file(url, filename):
    """Download a file from URL if it doesn't exist or is empty"""
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        print(f"Downloading {filename} from {url}...")
        try:
            response = requests.get(url, timeout=10)
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {filename}: {os.path.getsize(filename)} bytes")
            return True
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return False
    return True

def test_esp32_endpoints(base_url):
    """Test different ESP32-CAM endpoints to find the correct stream URL"""
    endpoints = [
        "",            # Root endpoint (/)
        "stream",      # Standard stream endpoint
        "mjpeg/1",     # Alternative stream endpoint
        "video",       # Another possible stream endpoint
        "capture"      # Single image endpoint
    ]
    
    working_endpoints = []
    
    for endpoint in endpoints:
        url = f"{base_url}{endpoint}"
        if url.endswith("/"):
            url = url[:-1]
        
        print(f"Testing endpoint: {url}")
        try:
            response = requests.get(url, stream=True, timeout=3)
            content_type = response.headers.get('content-type', '')
            
            if response.status_code == 200:
                print(f"✓ URL {url} accessible (Content-Type: {content_type})")
                if 'image/jpeg' in content_type or 'multipart/x-mixed-replace' in content_type:
                    print(f"✓ URL {url} appears to be a valid image/stream endpoint")
                    working_endpoints.append((url, content_type))
                # Close the connection
                response.close()
            else:
                print(f"✗ URL {url} returned status code {response.status_code}")
        except Exception as e:
            print(f"✗ URL {url} error: {e}")
    
    return working_endpoints

def process_frame(frame, net, output_layers, classes, colors, conf_threshold=0.3, nms_threshold=0.4):
    """Process a single frame with object detection"""
    if frame is None or frame.size == 0:
        return None
        
    height, width = frame.shape[:2]
    
    # Prepare image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    
    # Run inference
    net.setInput(blob)
    outputs = net.forward(output_layers)
    
    # Process detections
    boxes = []
    confidences = []
    class_ids = []
    
    # Process each output layer
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Filter detections by confidence threshold
            if confidence > conf_threshold:
                # Scale detection coordinates to original image size
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Calculate top-left corner of bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maximum suppression to remove overlapping bounding boxes
    idxs = []
    if len(boxes) > 0:
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    # Draw bounding boxes and labels
    if len(idxs) > 0:
        # For OpenCV 4.5.4+ compatibility
        if isinstance(idxs, np.ndarray):
            idxs = idxs.flatten()
        else:
            idxs = idxs.flatten() if len(idxs) > 0 else []
        
        for i in idxs:
            x, y, w, h = boxes[i]
            
            # Ensure box is within image boundaries
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)
            
            # Get class name safely
            class_id = class_ids[i]
            if class_id >= len(classes):
                class_name = "unknown"
            else:
                class_name = classes[class_id]
            
            # Create label with class name and confidence
            confidence = confidences[i]
            label = f"{class_name}: {confidence:.2f}"
            
            # Get color for class
            color = colors[class_id % len(colors)]
            # Convert from numpy array to tuple if necessary
            if isinstance(color, np.ndarray):
                color = tuple(map(int, color))
            
            # Draw rectangle and text
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Add label with background for better visibility
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x, y - 20), (x + text_size[0], y), color, -1)
            cv2.putText(frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Add FPS counter
    current_time = time.time()
    if hasattr(process_frame, 'last_time'):
        fps = 1 / (current_time - process_frame.last_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    process_frame.last_time = current_time
    
    # Add detection count
    detection_count = len(idxs)
    cv2.putText(frame, f"Detections: {detection_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

def stream_receiver(url, frame_buffer, stop_event):
    """Receives stream data in a separate thread"""
    print(f"Starting stream receiver thread for {url}")
    bytes_data = b''
    frames_received = 0
    reconnect_delay = 1
    
    while not stop_event.is_set():
        try:
            # Open stream with generous timeout
            stream = urllib.request.urlopen(url, timeout=20)
            print(f"Connected to stream at {url}")
            reconnect_delay = 1  # Reset delay on successful connection
            
            while not stop_event.is_set():
                chunk = stream.read(4096)  # Read larger chunks
                if not chunk:
                    print("Stream ended, reconnecting...")
                    break
                
                bytes_data += chunk
                
                # Find JPEG image in stream
                a = bytes_data.find(b'\xff\xd8')  # JPEG start
                b = bytes_data.find(b'\xff\xd9')  # JPEG end
                
                if a != -1 and b != -1:
                    # Extract JPEG image
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        # Update the frame buffer with new frame
                        frame_buffer["frame"] = frame.copy()
                        frames_received += 1
                        if frames_received % 10 == 0:
                            print(f"Received {frames_received} frames")
            
        except Exception as e:
            print(f"Stream error: {e}")
            if stop_event.is_set():
                break
                
            # Exponential backoff for reconnects (up to ~30 seconds)
            print(f"Reconnecting in {reconnect_delay} seconds...")
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30)
    
    print("Stream receiver thread stopped")

def capture_endpoint_handler(url, frame_buffer, stop_event):
    """Handles single image capture endpoint"""
    print(f"Starting capture endpoint handler for {url}")
    frames_received = 0
    
    while not stop_event.is_set():
        try:
            # Request a single image
            response = requests.get(url, timeout=5)
            if response.status_code == 200 and response.content:
                # Decode image
                frame = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    # Update the frame buffer with new frame
                    frame_buffer["frame"] = frame.copy()
                    frames_received += 1
                    if frames_received % 10 == 0:
                        print(f"Received {frames_received} frames")
            else:
                print(f"Capture failed with status {response.status_code}")
                
            # Short delay between captures
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Capture error: {e}")
            if stop_event.is_set():
                break
            time.sleep(1)
    
    print("Capture handler thread stopped")

def run_object_detection():
    """Main function to run object detection"""
    print("\n===== ESP32-CAM YOLOv4-Tiny Object Detection =====\n")
    
    # Check and download required YOLO files
    files = {
        "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
        "yolov4-tiny.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
        "coco.names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
    }
    
    # Download missing files
    for filename, url in files.items():
        if not download_file(url, filename):
            print(f"Failed to download {filename}. Please check your internet connection.")
            return
    
    # Load class names
    try:
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(classes)} object classes")
        if len(classes) == 0:
            raise ValueError("No classes loaded")
    except Exception as e:
        print(f"Error loading classes: {e}")
        return
    
    # Try to load YOLO model
    try:
        print("Loading YOLOv4-tiny model...")
        net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
        print("Model loaded successfully")
        
        # Check for GPU support
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Using CUDA for inference")
        except:
            print("CUDA not available, using CPU")
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        return
    
    # Get output layers
    layer_names = net.getLayerNames()
    try:
        # OpenCV 4.5.4+
        unconnected_layers = net.getUnconnectedOutLayers()
        if isinstance(unconnected_layers, np.ndarray):
            output_layers = [layer_names[i - 1] for i in unconnected_layers]
        else:
            output_layers = [layer_names[i[0] - 1] for i in unconnected_layers]
    except:
        # Older OpenCV versions
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    print(f"Using output layers: {output_layers}")
    
    # Random colors for class labels
    np.random.seed(42)  # For consistent colors
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    # Ask user for video source
    print("\nVideo Source Options:")
    print("1. ESP32-CAM Stream")
    print("2. Webcam")
    choice = input("Select video source (1 or 2, default is 1): ").strip() or "1"
    
    stop_event = threading.Event()
    frame_buffer = {"frame": None}  # Using dict as a mutable container
    
    if choice == "1":
        # ESP32-CAM stream
        ip_address = input("Enter ESP32-CAM IP address (default is 192.168.137.79): ").strip() or "192.168.137.79"
        base_url = f'http://{ip_address}/'
        
        print("\nTesting ESP32-CAM endpoints...")
        working_endpoints = test_esp32_endpoints(base_url)
        
        if not working_endpoints:
            print("No working endpoints found. Using default stream URL.")
            stream_url = f"{base_url}stream"
        else:
            print("\nWorking endpoints:")
            for i, (url, content_type) in enumerate(working_endpoints):
                print(f"{i+1}. {url} ({content_type})")
            
            endpoint_choice = input(f"Select endpoint (1-{len(working_endpoints)}, default is 1): ").strip() or "1"
            try:
                endpoint_idx = int(endpoint_choice) - 1
                if 0 <= endpoint_idx < len(working_endpoints):
                    stream_url = working_endpoints[endpoint_idx][0]
                else:
                    stream_url = working_endpoints[0][0]
            except:
                stream_url = working_endpoints[0][0]
        
        print(f"Using stream URL: {stream_url}")
        
        # Start stream receiver in a separate thread
        if "capture" in stream_url:
            # For single image capture endpoint
            receiver_thread = threading.Thread(target=capture_endpoint_handler, 
                                              args=(stream_url, frame_buffer, stop_event))
        else:
            # For MJPEG streaming endpoint
            receiver_thread = threading.Thread(target=stream_receiver, 
                                              args=(stream_url, frame_buffer, stop_event))
        
        receiver_thread.daemon = True
        receiver_thread.start()
        
        print("Processing stream in main thread. Press ESC to exit.")
        
        last_frame_time = time.time()
        frames_without_data = 0
        
        # Ask for confidence threshold
        conf_threshold_str = input("Enter detection confidence threshold (0.0-1.0, default is 0.3): ").strip() or "0.3"
        try:
            conf_threshold = float(conf_threshold_str)
            conf_threshold = max(0.1, min(0.9, conf_threshold))  # Clamp between 0.1 and 0.9
        except:
            conf_threshold = 0.3
        print(f"Using confidence threshold: {conf_threshold}")
        
        while not stop_event.is_set():
            frame = frame_buffer.get("frame")
            
            if frame is not None:
                # Process the frame with custom confidence threshold
                processed_frame = process_frame(frame.copy(), net, output_layers, classes, colors, 
                                               conf_threshold=conf_threshold, nms_threshold=0.4)
                if processed_frame is not None:
                    cv2.imshow("YOLOv4-Tiny Object Detection", processed_frame)
                    last_frame_time = time.time()
                    frames_without_data = 0
            else:
                frames_without_data += 1
                # Check for timeout (no frames received)
                if time.time() - last_frame_time > 30 or frames_without_data > 300:
                    print("No frames received for too long. Falling back to webcam.")
                    stop_event.set()
                    receiver_thread.join(timeout=1)
                    choice = "2"  # Fall back to webcam
                    break
            
            # Check for ESC key
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                print("ESC pressed. Exiting...")
                stop_event.set()
                receiver_thread.join(timeout=1)
                break
    
    if choice == "2":
        # Webcam
        print("\nOpening webcam...")
        
        # Try different camera indices
        cam_found = False
        for camera_id in range(3):  # Try indices 0, 1, 2
            print(f"Trying webcam {camera_id}...")
            cap = cv2.VideoCapture(camera_id)
            
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"Webcam {camera_id} opened successfully")
                    cam_found = True
                    break
                else:
                    cap.release()
                    print(f"Webcam {camera_id} opened but couldn't read frames")
            else:
                print(f"Failed to open webcam {camera_id}")
        
        # Check if any webcam was opened
        if not cam_found:
            print("Failed to open any webcam. Exiting.")
            return
        
        # Set webcam resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Ask for confidence threshold
        conf_threshold_str = input("Enter detection confidence threshold (0.0-1.0, default is 0.3): ").strip() or "0.3"
        try:
            conf_threshold = float(conf_threshold_str)
            conf_threshold = max(0.1, min(0.9, conf_threshold))  # Clamp between 0.1 and 0.9
        except:
            conf_threshold = 0.3
        print(f"Using confidence threshold: {conf_threshold}")
        
        try:
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Failed to get frame from webcam")
                    break
                
                # Process and display frame
                processed_frame = process_frame(frame, net, output_layers, classes, colors, 
                                              conf_threshold=conf_threshold, nms_threshold=0.4)
                if processed_frame is not None:
                    cv2.imshow("YOLOv4-Tiny Object Detection", processed_frame)
                
                # Check for ESC key or 'q'
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or 'q'
                    print("Exit key pressed. Exiting...")
                    break
        finally:
            cap.release()
    
    cv2.destroyAllWindows()
    print("\nApplication closed")
    print("\n===== Thank you for using ESP32-CAM YOLOv4-Tiny Object Detection =====")

if __name__ == "__main__":
    # Ensure requests library is installed
    try:
        import requests
    except ImportError:
        print("Installing requests library...")
        try:
            import pip
            pip.main(['install', 'requests'])
            import requests
        except Exception as e:
            print(f"Failed to install requests: {e}")
            print("Please install the requests library manually with:")
            print("pip install requests")
            exit(1)
    
    try:
        run_object_detection()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("Please check your setup and try again")