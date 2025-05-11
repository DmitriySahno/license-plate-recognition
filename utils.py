import cv2
import numpy as np

VEHICLES = [2, 3, 5, 7]     # COCO classification {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
LICENSE_PLATE_PATTERN = "AAANNNNA"
# Make sure both letters and digits are included
CUSTOM_CHAR_LIST = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

char_to_digit = {
    'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'E': '3', 'A': '4',
    'S': '5', 'G': '6', 'T': '7', 'B': '8', 'P': '9'
}

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

    def normalize_license_plate(self, img):
        if img is None:
            print("No image provided for license plate normalization.")
            return None
        
        print(f"Normalizing license plate...")
        result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, result = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        return result


    def normalize_license_number(self, text):
        # regex_patterns = LICENSE_PLATE_PATTERN.replace('A', '[A-Za-z]').replace('N', '\\d')
        result = ""
        for i, c in enumerate(text):
            if i < len(LICENSE_PLATE_PATTERN):
                if LICENSE_PLATE_PATTERN[i] == 'A' and c.isdigit():
                    result += digit_to_char.get(c, c)
                elif LICENSE_PLATE_PATTERN[i] == 'N' and c.isalpha():
                    result += char_to_digit.get(c.upper(), c)
                else:
                    result += c
            else:
                result += c
        
        return result.upper()
    

    def extract_license_plate_number(self, img, use_upsample=False):
        if img is None:
            print("No image provided for license plate number extraction.")
            return None
        
        print("Extracting license plate number...")
        if use_upsample and self.upscale_model:
            img = self.upscale_model.predict(img)[0]
        
        # result = self.reader.readtext(img, detail=0, paragraph=True, allowlist=CUSTOM_CHAR_LIST) #easyocr
        result = self.ocr.ocr(img, cls=True)
        if result is None or len(result) == 0 or result[0] is None:
            print("No text detected.")
            return None
        
        text = "".join([item[1][0] for sublist in result for item in sublist])
        print(f"License plate: {text}")
        result = self.normalize_license_number(text)
        return result

    def upscale_license_plate(self, img):
        w, h, _ = img.shape
        if w < 100:
            print(f"Upscaling license plate... {w}x{h}")
            # img, _ = upscale_model.enhance(img, outscale=2)
            img = self.upscale_model.upsample(img)
        else:
            print(f"License plate upscaling is not necessary... {w}x{h}")