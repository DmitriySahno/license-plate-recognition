# License Plate Recognition

## Overview
This project detects vehicles, extracts license plates, and recognizes license plate numbers from video streams using YOLO, PaddleOCR, and SORT.
Currently uses detection area for recognition the target of camera

## Features
- Real-time vehicle and license plate detection:
    - Vehicle detection with yolo11s
    - License plate detection inside vehicle bbox with custom NN in logs/retrain/weights/best.pt, accuracy 98%
- License plate number recognition using PaddleOCR.
- Video processing with overlays for detected objects.
- Video demonstration is in output path

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/license-plate-recognition.git
   cd license-plate-recognition
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
- Run the main script:
    ```bash
    python [main.py](http://_vscodecontentref_/14)
    ```
- Process a video file:
Use the video_converter.ipynb notebook to process videos and save the output.

- Train the YOLO model:
Use the training.ipynb notebook to retrain the YOLO model with a custom dataset.

## File Structure
* `main.py`: Main script for real-time license plate recognition.
* `utils.py`: Utility functions for detection, OCR, and visualization.
* `video_converter.ipynb`: Notebook for processing video files.
* `training.ipynb`: Notebook for training the YOLO model with custom license plate.
* `models/`: Pre-trained models, used in project.
* `source/`: Input video files.
* `output/`: Processed video files with video_converter.ipynb.
* `requirements.txt`: Python dependencies.