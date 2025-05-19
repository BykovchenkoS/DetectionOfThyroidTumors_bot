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
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, num_classes)

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        return model

    def _transform(self, image):
        transform = torchvision.transforms.ToTensor()
        return transform(image)

    def process_image(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = self._transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                predictions = self.model(img_tensor)

            result_image = self._visualize_predictions(img, predictions[0])
            from io import BytesIO
            buf = BytesIO()
            result_image.save(buf, format='PNG')
            buf.seek(0)
            return buf

        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def _visualize_predictions(self, image, predictions, threshold=0.5):
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        masks = predictions['masks'].cpu().numpy()

        keep = scores >= threshold
        boxes = boxes[keep]
        labels = labels[keep]
        masks = masks[keep]

        color_map = {
            1: [0.4, 0, 0.4],
            2: [0, 1, 0],
        }

        img_array = np.array(image).astype(np.float32) / 255.0

        for box, label, mask in zip(boxes, labels, masks):
            if label not in color_map:
                continue

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
