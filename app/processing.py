# app/processing.py
import cv2
import os
import re
import tempfile
import unicodedata
from typing import Optional
from ultralytics import YOLO
from .inf_video import predict_frames_in_directory

def blurred_frame_differencing(frame1, frame2):
    blurred_frame1 = cv2.GaussianBlur(frame1, (5, 5), 0)
    blurred_frame2 = cv2.GaussianBlur(frame2, (5, 5), 0)
    return cv2.absdiff(blurred_frame1, blurred_frame2)

def secure_filename(filename: str) -> str:
    filename = unicodedata.normalize("NFKD", filename).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9_.-]", "", filename)

def _extract_frames_gray(video_path: str, output_dir: str, frame_skip: int = 5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    count = 0
    os.makedirs(output_dir, exist_ok=True)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None:
                continue
            if count % frame_skip == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(os.path.join(output_dir, f'frame_{count}.jpg'), gray)
            count += 1
    finally:
        cap.release()

def _process_frame(model: YOLO, frame_path: str, roi_dir: str):
    img = cv2.imread(frame_path)
    results = model(img)
    boxes = results[0].boxes.xyxy.tolist() if results and results[0].boxes is not None else []
    for box in boxes:
        x1, y1, x2, y2 = [int(c) for c in box]
        crop = img[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(roi_dir, os.path.basename(frame_path)), crop)

def _extract_nostrils_from_frames(frames_dir: str, roi_dir: str, detector: YOLO):
    os.makedirs(roi_dir, exist_ok=True)
    for f in os.listdir(frames_dir):
        if f.endswith(".jpg"):
            _process_frame(detector, os.path.join(frames_dir, f), roi_dir)
    return roi_dir

def _subtract_adjacent(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    def key(f): 
        m = re.findall(r'\d+', f)
        return int(m[0]) if m else 0
    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")], key=key)

    for i in range(len(files) - 1):
        f1 = cv2.imread(os.path.join(input_dir, files[i]))
        f2 = cv2.imread(os.path.join(input_dir, files[i+1]))
        if f1 is None or f2 is None:
            continue
        if f1.shape != f2.shape:
            # resize smaller to bigger
            if f1.size < f2.size:
                f1 = cv2.resize(f1, (f2.shape[1], f2.shape[0]))
            else:
                f2 = cv2.resize(f2, (f1.shape[1], f1.shape[0]))
        sub = blurred_frame_differencing(f1, f2)
        cv2.imwrite(os.path.join(output_dir, f'frame_{i}.jpg'), sub)
    return output_dir

def predict_from_video(
    video_path: str,
    mode: str,
    *,
    nostrils_model_path: str,
    abdomen_model_path: str,
    yolo_weights_path: Optional[str] = None
) -> str:
    """
    mode in {"nostrils", "abdomen"}
    """
    with tempfile.TemporaryDirectory() as tmp:
        frames = os.path.join(tmp, "frames")
        _extract_frames_gray(video_path, frames)

        if mode == "nostrils":
            if not yolo_weights_path:
                raise ValueError("YOLO weights path required for nostrils mode.")
            detector = YOLO(yolo_weights_path)
            roi_dir = _extract_nostrils_from_frames(frames, os.path.join(tmp, "roi"), detector)
            sub_dir = _subtract_adjacent(roi_dir, os.path.join(tmp, "sub"))
            return predict_frames_in_directory(sub_dir, nostrils_model_path)
        elif mode == "abdomen":
            sub_dir = _subtract_adjacent(frames, os.path.join(tmp, "sub"))
            return predict_frames_in_directory(sub_dir, abdomen_model_path)
        else:
            raise ValueError("Invalid mode")
