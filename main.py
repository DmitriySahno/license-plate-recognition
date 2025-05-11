import cv2
from paddleocr import PaddleOCR
from ultralytics import YOLO
import numpy as np
from utils import Utils


# init models
coco_model = YOLO("models/yolo11s.pt")
license_plate_model = YOLO("logs/retrain/weights/best.pt")
ocr = PaddleOCR(use_angle_cls=True, lang="en")

utils = Utils(license_plate_model, coco_model, ocr, None)

video = cv2.VideoCapture("source/video1.mp4")
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
detecting_area = (width//4, height//4, width*3//4, height*3//4)

overlay = np.zeros((height, width, 3), dtype=np.uint8)
cv2.rectangle(overlay, (detecting_area[0], detecting_area[1]), (detecting_area[2], detecting_area[3]), (0, 255, 0), -1)

while True:
    ret, frame = video.read()
    
    if not ret:
        break
    
    img = frame[detecting_area[1]:detecting_area[3], detecting_area[0]:detecting_area[2]]
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    coco_results = coco_model.predict(img, conf=0.5)
    if not coco_results:
        continue
    
    for vehicle in (object for object in coco_results[0].boxes.data.tolist() if utils.is_vehicle(object[-1])):    
        vehicle_img = utils.crop_vehicle(img, vehicle)
        
        license_plate_obj = utils.fetch_license_plate(vehicle_img)
        if license_plate_obj is not None:
            license_plate_img, x, y = license_plate_obj
            license_plate_number = utils.extract_license_plate_number(license_plate_img)

            x = x + int(vehicle[0])
            y = y + int(vehicle[1])
        
            plate_h, plate_w, _ = license_plate_img.shape
            
            text_size, _ = cv2.getTextSize(license_plate_number, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_w, text_h = text_size
            cv2.rectangle(frame, (0, 0), (0 + text_w, 0 + text_h), (0, 0, 0), -1)
            cv2.putText(frame, license_plate_number, (0, text_h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            

    cv2.rectangle(frame, (detecting_area[0], detecting_area[1]), (detecting_area[2], detecting_area[3]), (0, 255, 0), 2)
    frame = cv2.addWeighted(overlay, 0.2, frame, 1, 0)
    
    cv2.imshow("Vehicle", frame)
    
    # Wait for 1ms for key press to continue or exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
video.release()