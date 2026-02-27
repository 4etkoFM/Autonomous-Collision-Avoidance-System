import cv2
import os
from ultralytics import YOLO

IMAGES_DIR = 'train/images'
MODEL_PATH = 'best.pt'
CONF_THRES = 0.25

model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names

image_files = [
    f for f in os.listdir(IMAGES_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

if not image_files:
    raise RuntimeError('В папке нет изображений!')

print(f'Найдено изображений: {len(image_files)}')

for img_name in image_files:
    img_path = os.path.join(IMAGES_DIR, img_name)

    results = model(img_path, conf=CONF_THRES)

    img = cv2.imread(img_path)

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            class_name = CLASS_NAMES[cls_id]
            label = f'{class_name} {conf:.2f}'

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    cv2.imshow('YOLOv8 Detection', img)
    key = cv2.waitKey(0)

    if key == 27:
        break


cv2.destroyAllWindows()
