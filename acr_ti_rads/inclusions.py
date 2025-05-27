import cv2
import numpy as np
from database import db
from pathlib import Path


def analyze_inclusions(image, mask):
    if image is None or mask is None:
        raise ValueError("Изображение или маска не были загружены корректно")

    node_region = cv2.bitwise_and(image, image, mask=mask)

    _, thresholded = cv2.threshold(node_region, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    inclusion_types = []
    max_points = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 10:
            continue

        perimeter = cv2.arcLength(contour, True)
        compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else float('inf')

        mask_inclusion = np.zeros_like(image)
        cv2.drawContours(mask_inclusion, [contour], -1, 255, thickness=cv2.FILLED)
        inclusion_pixels = image[mask_inclusion > 0]
        mean_intensity = np.mean(inclusion_pixels)

        if area > 1000 and compactness < 1.5:
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull)
            if defects is not None and len(defects) > 2:
                inclusion_type = "Крупный артефакт с 'хвостом кометы'"
                points = 0
            else:
                inclusion_type = "Макрокальцинаты"
                points = 1
        elif area > 100 and compactness < 1.8:
            edge_distance = cv2.pointPolygonTest(contour, (image.shape[1] // 2, image.shape[0] // 2), True)
            if edge_distance < 0:
                inclusion_type = "Периферическая кальцификация"
                points = 2
            else:
                inclusion_type = "Точечные эхогенные очаги"
                points = 3
        elif area > 50 and mean_intensity > 200:
            inclusion_type = "Точечные эхогенные очаги"
            points = 3
        else:
            continue

        inclusion_types.append(inclusion_type)
        max_points = max(max_points, points)

    if not inclusion_types:
        inclusion_types.append("Нет")
        max_points = 0

    return {
        "types": list(set(inclusion_types)),
        "points": max_points
    }


def get_tirads_inclusion_info(inclusion_type_list):
    mapping = {
        "Крупный артефакт с 'хвостом кометы'": "Крупный артефакт \"хвост кометы\"",
        "Макрокальцинаты": "Макрокальцинаты",
        "Периферическая кальцификация": "Периферическая кальцификация",
        "Точечные эхогенные очаги": "Точечные эхогенные очаги",
        "Нет": "Нет или крупный артефакт \"хвост кометы\""
    }

    results = []

    for inc_type in inclusion_type_list:
        db_option_name = mapping.get(inc_type)
        if not db_option_name:
            continue

        query = """
            SELECT * FROM tirads_options 
            WHERE category = 'inclusions' AND option_name = %s
        """
        result = db.fetch_one(query, (db_option_name,))
        if result:
            results.append(result)

    return results


if __name__ == "__main__":
    image_path = '../cropped_regions/thyroid_carotis_597102879_20250527_143045_20250527_143047.jpg'
    mask_path = '../sam_predictions/binary_masks/thyroid_carotis_597102879_20250527_143045_20250527_143047_binary_mask_0.png'

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print("Ошибка: Не удалось загрузить изображение или маску.")
    else:
        inclusion_data = analyze_inclusions(image, mask)
        print(f"Включения: {inclusion_data}")

        tirads_data = get_tirads_inclusion_info(inclusion_data['types'])
        print(f"TIRADS информация: {tirads_data}")
