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
        self.yolo = YOLO(yolo_weights_path)

        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
        self.sam.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        checkpoint = torch.load(sam_finetuned_path, map_location='cpu')
        self.sam.mask_decoder.load_state_dict(checkpoint['mask_decoder_state_dict'])
        self.sam.eval()

        self.output_dir = "sam_predictions"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "binary_masks"), exist_ok=True)

    def _save_binary_mask(self, mask, image_path, suffix=""):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        mask_dir = os.path.join(self.output_dir, "binary_masks")
        os.makedirs(mask_dir, exist_ok=True)

        filename = f"{base_name}_binary_mask{suffix}.png"
        save_path = os.path.join(mask_dir, filename)

        binary_mask = (mask * 255).astype(np.uint8)
        cv2.imwrite(save_path, binary_mask)

        print(f"[DEBUG] Бинарная маска сохранена: {save_path}")
        return save_path

    def run_yolo_on_image(self, image_path):
        results = self.yolo(image_path)
        boxes = []

        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                cls_id = int(box.cls.item())
                conf = box.conf.item()

                class_name = 'Node'

                boxes.append({
                    'bbox': xyxy,
                    'cls': cls_id,
                    'class_name': class_name,
                    'conf': conf
                })

        return boxes

    def run_sam_with_boxes(self, image_path, boxes):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_for_drawing = image_rgb.copy()

        sam_predictor = SamPredictor(self.sam)
        sam_predictor.set_image(image_rgb)

        all_masks = []

        for i, box in enumerate(boxes):
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

            self._save_binary_mask(binary_mask, image_path, suffix=f"_{i}")

            mask_overlay = np.zeros_like(image_for_drawing)
            mask_overlay[binary_mask == 1] = [255, 0, 0]
            image_for_drawing = cv2.addWeighted(image_for_drawing, 1.0, mask_overlay, 0.5, 0)

            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image_for_drawing, (x1, y1), (x2, y2), (255, 0, 0), 1)

            label_text = f"{box['class_name']}"
            text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

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
        output_path = os.path.join(self.output_dir, f"{base_name}_mask.png")
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
            return None, None, None

        sorted_boxes = sorted(boxes, key=lambda x: x['conf'], reverse=True)

        best_box = sorted_boxes[0]

        print(f"[DEBUG] Найдено {len(boxes)} узлов. Используется самый уверенный.")

        print(f"[DEBUG] Точная сегментация через SAM...")
        masks, mask_vis_path = self.run_sam_with_boxes(cropped_image_path, [best_box])

        binary_mask_path = self._save_binary_mask(masks[0], cropped_image_path, suffix="_0")

        return masks, mask_vis_path, binary_mask_path
