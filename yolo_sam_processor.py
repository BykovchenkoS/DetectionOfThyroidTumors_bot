import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor


class YOLOSAMNodeAnalyzer:
    def __init__(self, yolo_weights_path, sam_checkpoint_path, sam_finetuned_path, model_type="vit_h"):
        # Инициализируем YOLO
        self.yolo = YOLO(yolo_weights_path)

        # Инициализируем SAM
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
        self.sam.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Загружаем дообученные веса SAM
        checkpoint = torch.load(sam_finetuned_path, map_location='cpu')
        self.sam.mask_decoder.load_state_dict(checkpoint['mask_decoder_state_dict'])
        self.sam.eval()

        # Папки
        self.output_dir = "sam_predictions"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "masks"), exist_ok=True)

    def run_yolo_on_image(self, image_path):
        """Запуск YOLO и получение bounding box с именами классов"""
        results = self.yolo(image_path)
        boxes = []

        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                cls_id = int(box.cls.item())
                conf = box.conf.item()

                # Используем имя класса вместо цифры
                class_name = 'Node'  # если только один класс

                boxes.append({
                    'bbox': xyxy,
                    'cls': cls_id,
                    'class_name': class_name,
                    'conf': conf
                })

        return boxes

    def run_sam_with_boxes(self, image_path, boxes):
        """Запуск SAM с боксами из YOLO и визуализация с красной маской и подписью 'Node'"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_for_drawing = image_rgb.copy()

        sam_predictor = SamPredictor(self.sam)
        sam_predictor.set_image(image_rgb)

        all_masks = []

        for box in boxes:
            bbox = box['bbox']
            input_box = np.array(bbox)

            masks, _, _ = sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False
            )

            binary_mask = masks[0].astype(np.uint8)
            all_masks.append(binary_mask)

            # Рисуем красную полупрозрачную маску
            mask_overlay = np.zeros_like(image_for_drawing)
            mask_overlay[binary_mask == 1] = [255, 0, 0]  # красная маска
            image_for_drawing = cv2.addWeighted(image_for_drawing, 1.0, mask_overlay, 0.5, 0)

            # Рисуем бокс
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image_for_drawing, (x1, y1), (x2, y2), (255, 0, 0), 1)  # красный бокс

            # Подпись над боксом
            label_text = f"{box['class_name']}"
            text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Позиция текста
            text_x = x1
            text_y = y1 - 5

            if text_y < 0:
                text_y = y2 + 15

            cv2.rectangle(
                image_for_drawing,
                (text_x, text_y - text_size[1] - 2),
                (text_x + text_size[0], text_y + 2),
                (255, 255, 255),
                thickness=cv2.FILLED
            )

            cv2.putText(
                image_for_drawing,
                label_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = "sam_predictions"
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"{base_name}_mask.png")
        plt.figure(figsize=(10, 10))
        plt.imshow(image_for_drawing)
        plt.axis("off")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        return all_masks, output_path

    def process(self, cropped_image_path):
        print(f"[DEBUG] Обнаружение узла через YOLO...")
        boxes = self.run_yolo_on_image(cropped_image_path)

        if not boxes:
            print("[INFO] Узлы не найдены YOLO.")
            return None, None

        print(f"[DEBUG] Найдено {len(boxes)} узлов.")

        print(f"[DEBUG] Точная сегментация через SAM...")
        masks, mask_vis_path = self.run_sam_with_boxes(cropped_image_path, boxes)

        return masks, mask_vis_path
