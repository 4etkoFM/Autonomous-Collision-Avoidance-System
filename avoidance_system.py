import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os

model = YOLO('runs/detect/train2/weights/best.pt')
CLASS_NAMES = model.names
DANGER_CLASSES = [
    'large_debris',
    'medium_debris',
    'small_debris',
    'rocket',
    'satellite'
]

def put_russian_text(img, text, position, font_size=24, color=(0, 255, 0)):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def calculate_avoidance_maneuver(frame, results, danger_center_ratio=0.2):
    h, w = frame.shape[:2]
    center_x = w // 2
    danger_margin = int(w * danger_center_ratio / 2)
    danger_left = center_x - danger_margin
    danger_right = center_x + danger_margin
    obstacles_in_danger_zone = []

    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = CLASS_NAMES.get(class_id, 'unknown')
            if class_name not in DANGER_CLASSES:
                continue
            obj_bottom_center_x = (x1 + x2) // 2
            obj_bottom_y = y2
            if danger_left <= obj_bottom_center_x <= danger_right:
                side = 'left' if obj_bottom_center_x < center_x else 'right'
                obstacles_in_danger_zone.append((side, obj_bottom_center_x, obj_bottom_y))

    if not obstacles_in_danger_zone:
        return "Курс чист. Движение прямо."
    left_obstacles = [obj for obj in obstacles_in_danger_zone if obj[0] == 'left']
    right_obstacles = [obj for obj in obstacles_in_danger_zone if obj[0] == 'right']
    if len(left_obstacles) < len(right_obstacles):
        maneuver = "Сместиться влево."
    elif len(right_obstacles) < len(left_obstacles):
        maneuver = "Сместиться вправо."
    else:
        avg_left_y = np.mean([obj[2] for obj in left_obstacles]) if left_obstacles else h
        avg_right_y = np.mean([obj[2] for obj in right_obstacles]) if right_obstacles else h
        maneuver = "Сместиться влево." if avg_left_y > avg_right_y else "Сместиться вправо."
    return maneuver

def run_avoidance_system(video_source=0):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Ошибка открытия источника: {video_source}")
        return
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        results = model(frame, conf=0.3)
        command = calculate_avoidance_maneuver(frame, results)
        frame = put_russian_text(frame, command, (30, 30))
        cv2.imshow('Autonomous Collision Avoidance System', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Работа завершена")

if __name__ == "__main__":
    run_avoidance_system(video_source=0)