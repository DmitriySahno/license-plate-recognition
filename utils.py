import cv2
import numpy as np
from datetime import datetime

VEHICLES = [2, 3, 5, 7]     # COCO classification {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
LICENSE_PLATE_PATTERN = "AAANNNNA"

char_to_digit = {'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'E': '3', 'A': '4', 'S': '5', 'G': '6', 'T': '7', 'B': '8', 'P': '9'}

digit_to_char = {v: k for k, v in char_to_digit.items()}

class Utils:
    def __init__(self, license_plate_model, coco_model, ocr, upscale_model=None):
        self.license_plate_model = license_plate_model
        self.coco_model = coco_model
        self.ocr = ocr
        self.upscale_model = upscale_model
        
    def is_vehicle(self, class_id):
        return class_id in VEHICLES

    def crop_vehicle(self, img, vehicle_data):
        if img is None:
            print("No image provided for vehicle cropping.")
            return None
        
        print("Cropping vehicle...")
        x1, y1, x2, y2 = np.array(vehicle_data[:4], dtype=int)
        return np.array(img[y1:y2, x1:x2])

    def fetch_license_plate(self, img):
        if img is None:
            print("No image provided for license plate cropping.")
            return None
        
        print("Detecting license plate...")
        license_plate_results = self.license_plate_model.predict(img)[0]
        
        if len(license_plate_results.boxes) == 0:
            print("No license plate detected.")
            return None

        x1, y1, x2, y2 = np.array(license_plate_results.boxes[0].xyxy[0].cpu(), dtype=int)
        return img[y1:y2, x1:x2], x1, y1

    def normalize_license_number(self, text):
        result = ""
        
        if len(text) != len(LICENSE_PLATE_PATTERN):
            print(f"Invalid license plate number: {text}")
            return None
        
        for idx, char in enumerate(text):
            if idx < len(LICENSE_PLATE_PATTERN):
                if LICENSE_PLATE_PATTERN[idx] == 'A' and char.isdigit():
                    result += digit_to_char.get(char, char)
                elif LICENSE_PLATE_PATTERN[idx] == 'N' and char.isalpha():
                    result += char_to_digit.get(char.upper(), char)
                else:
                    result += char
            else:
                result += char
        
        return result.upper()
    
    def extract_license_plate_number(self, img):
        if img is None:
            print("No image provided for license plate number extraction.")
            return None
        
        print("Extracting license plate number...")
        try:
            result = self.ocr.ocr(np.array(img), cls=True)
        except Exception as e:
            print(f"Error during OCR: {e}")
            return None
            
        if result is None or len(result) == 0 or result[0] is None:
            print("No text detected.")
            return None
        
        text = "".join([item[1][0] for sublist in result for item in sublist])
        print(f"License plate: {text}, ocr result: {result}")
        result = self.normalize_license_number(text)
        return result
    
    def draw_license_plate_number(self, license_plate_number, frame, position):
        text_size, _ = cv2.getTextSize(license_plate_number, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_w, text_h = text_size
        x, y = position
        cv2.rectangle(frame, (x, y), (x + text_w, y + text_h), (0, 0, 0), -1)
        cv2.putText(frame, license_plate_number, (x, text_h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
