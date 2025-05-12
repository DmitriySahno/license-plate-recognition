import cv2
from paddleocr import PaddleOCR
from ultralytics import YOLO
import numpy as np
from utils import Utils
from sort.sort import Sort


# init models
coco_model = YOLO("models/yolo11s.pt")
license_plate_model = YOLO("logs/retrain/weights/best.pt")
ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)

utils = Utils(license_plate_model, coco_model, ocr, None)

video = cv2.VideoCapture("source/video1.mp4")
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
detecting_area = (width//4, height//4, width*3//4, height*3//4)

overlay = np.zeros((height, width, 3), dtype=np.uint8)
cv2.rectangle(overlay, (detecting_area[0], detecting_area[1]), (detecting_area[2], detecting_area[3]), (0, 255, 0), -1)

tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)

tracked_objs = []
tracked_ids = []
license_plate_number = None

each_nth_frame = 1
frame_number = -1
while True:
    frame_number += 1
    ret, frame = video.read()
    
    # each nth frame
    if (frame_number % each_nth_frame != 0):
        continue
    
    if not ret:
        break
    
    img = frame[detecting_area[1]:detecting_area[3], detecting_area[0]:detecting_area[2]]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    coco_results = coco_model.predict(img, conf=0.5)
    if not coco_results:
        continue
    
    detections = [
        [x1, y1, x2, y2, score]
        for x1, y1, x2, y2, score, cls in coco_results[0].boxes.data.tolist()
        if utils.is_vehicle(cls)
    ]
    
    detections = np.array(detections) if detections else np.empty((0, 5))
    tracked_objs = tracker.update(detections)
    
    new_ids = tracked_objs[:, -1]
    if set(new_ids) != set(tracked_ids) or len(tracked_ids) > 0 and license_plate_number is None:
        tracked_ids = new_ids
        print("Detecting IDs: ", tracked_ids)
        license_plate_number = None
        
        for vehicle in tracked_objs:
            print("Vehicle detected: ", vehicle)
            vehicle_img = utils.crop_vehicle(img, vehicle)
            
            license_plate_obj = utils.fetch_license_plate(vehicle_img)
            if license_plate_obj is not None:
                license_plate_img, x, y = license_plate_obj
                license_plate_number = utils.extract_license_plate_number(license_plate_img)
                
    utils.draw_license_plate_number(license_plate_number, frame, (0, 0))
    cv2.rectangle(frame, (detecting_area[0], detecting_area[1]), (detecting_area[2], detecting_area[3]), (0, 255, 0), 2)
    frame = cv2.addWeighted(overlay, 0.1, frame, 1, 0)
    
    cv2.imshow("License Plate Recognition", frame)
    
    # Wait for 1ms for key press to continue or exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
video.release()