import os
from datetime import datetime
from io import BytesIO
import cv2
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np


class MaskRCNNThyroidAnalyzer:
    def __init__(self, model_path):
        self.class_names = ['background', 'Thyroid tissue', 'Carotis']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        model = maskrcnn_resnet50_fpn_v2(weights=None)

        num_classes = len(self.class_names)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features,
                                                                                                   num_classes)

        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256,
                                                                                                  num_classes)

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        return model

    def _transform(self, image):
        transform = torchvision.transforms.ToTensor()
        return transform(image)

    def _save_binary_mask(self, mask, class_name, base_path):
        mask_dir = os.path.join(os.path.dirname(base_path), 'binary_masks')
        os.makedirs(mask_dir, exist_ok=True)

        filename = f"{os.path.splitext(os.path.basename(base_path))[0]}_{class_name.replace(' ', '_')}_mask.png"
        save_path = os.path.join(mask_dir, filename)

        binary_mask = (mask * 255).astype(np.uint8)
        mask_image = Image.fromarray(binary_mask)
        mask_image.save(save_path)

        return save_path

    def _crop_combined_thyroid_carotis(self, pil_image, predictions, image_path):
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        masks = predictions['masks'].cpu().numpy()
        keep = scores >= 0.5

        boxes = boxes[keep]
        labels = labels[keep]
        masks = masks[keep]

        all_coords = []
        for box, label, mask in zip(boxes, labels, masks):
            if self.class_names[label] in ['Thyroid tissue', 'Carotis']:
                x1, y1, x2, y2 = map(int, box)
                all_coords.append((x1, y1, x2, y2))

                class_name = self.class_names[label]
                binary_mask = (mask[0] > 0.5).astype(np.uint8)
                self._save_binary_mask(binary_mask, class_name, image_path)

        if not all_coords:
            print("Не найдены объекты Thyroid tissue или Carotis")
            return None

        min_x = min(box[0] for box in all_coords)
        min_y = min(box[1] for box in all_coords)
        max_x = max(box[2] for box in all_coords)
        max_y = max(box[3] for box in all_coords)

        img_array = np.array(pil_image)
        cropped_img = img_array[min_y:max_y, min_x:max_x]

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = 'cropped_regions'
        os.makedirs(output_dir, exist_ok=True)

        save_path = os.path.join(output_dir, f"thyroid_carotis_{base_name}_{timestamp}.jpg")

        cropped_pil = Image.fromarray(cropped_img)
        cropped_pil.save(save_path)

        return save_path

    def process_image(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = self._transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                predictions = self.model(img_tensor)

            prediction_dict = predictions[0]
            result_image = self._visualize_predictions(img, prediction_dict)

            buf = BytesIO()
            result_image.save(buf, format='PNG')
            buf.seek(0)
            combined_cropped_path = self._crop_combined_thyroid_carotis(img, prediction_dict, image_path)

            return buf, prediction_dict, combined_cropped_path

        except Exception as e:
            print(f"Error processing image: {e}")
            return None, None, None

    def _visualize_predictions(self, image, predictions, threshold=0.5):
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        masks = predictions['masks'].cpu().numpy()

        keep = scores >= threshold
        boxes = boxes[keep]
        labels = labels[keep]
        masks = masks[keep]

        if len(boxes) == 0:
            print("Ничего не найдено")
            return image

        img_array = np.array(image).astype(np.float32) / 255.0
        color_map = {
            1: [0.4, 0, 0.4],
            2: [0, 1, 0],
        }

        displayed_classes = set()

        for score, box, label, mask in sorted(zip(scores, boxes, labels, masks), key=lambda x: x[0], reverse=True):
            if label in displayed_classes or label not in color_map:
                continue

            displayed_classes.add(label)
            color = color_map[label]
            class_name = self.class_names[label]

            mask = mask[0] > 0.5
            img_array[mask] = 0.5 * img_array[mask] + 0.5 * np.array(color)

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)

            text_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_bg_top_left = (x1, y1 - text_size[1] - 5)
            text_bg_bottom_right = (x1 + text_size[0], y1 - 5)

            cv2.rectangle(
                img_array,
                text_bg_top_left,
                text_bg_bottom_right,
                (1, 1, 1),
                thickness=cv2.FILLED
            )

            text_color = tuple(map(lambda x: int(x * 255), color))
            cv2.putText(
                img_array,
                class_name,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                1
            )

        result_img = Image.fromarray((img_array * 255).astype(np.uint8))

        return result_img
