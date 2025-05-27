import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import os
from database import db


def analyze_inclusions(image, mask):
    node_region = cv2.bitwise_and(image, image, mask=mask)
    _, thresholded = cv2.threshold(node_region, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return ["Нет"]

    inclusion_types = []

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if area < 10:
            continue

        perimeter = cv2.arcLength(contour, True)
        compactness = (perimeter ** 2) / (4 * np.pi * area)
        mask_inclusion = np.zeros_like(image)
        cv2.drawContours(mask_inclusion, [contour], -1, 255, thickness=cv2.FILLED)
        inclusion_pixels = image[mask_inclusion > 0]
        mean_intensity = np.mean(inclusion_pixels)

        if area > 1000 and compactness < 1.5:
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull)
            if defects is not None and len(defects) > 2:
                inclusion_types.append("Крупный артефакт с 'хвостом кометы'")
            else:
                inclusion_types.append("Макрокальцинаты")
        elif area > 100 and compactness < 1.8:
            edge_distance = cv2.pointPolygonTest(contour, (image.shape[1] // 2, image.shape[0] // 2), True)
            if edge_distance < 0:
                inclusion_types.append("Периферическая кальцификация")
            else:
                inclusion_types.append("Точечные эхогенные очаги")
        elif area > 50 and mean_intensity > 200:
            inclusion_types.append("Точечные эхогенные очаги")

    unique_inclusions = list(set(inclusion_types)) if inclusion_types else ["Нет"]
    return unique_inclusions


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


def process_custom_image(image_path, mask_path):
    if not os.path.exists(image_path):
        print(f"Файл изображения не найден: {image_path}")
        return
    if not os.path.exists(mask_path):
        print(f"Файл маски не найден: {mask_path}")
        return

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    inclusion_types = analyze_inclusions(image, mask)
    tirads_data_list = get_tirads_inclusion_info(inclusion_types)

    print("\nРезультат анализа включений:")
    print(f"Типы включений: {inclusion_types}")

    if tirads_data_list:
        for data in tirads_data_list:
            print(f"\nСоответствует TIRADS опции: {data['option_name']}")
            print(f"Баллы: {data['points']}")
            print(f"Описание: {data['description']}")
    else:
        print("Не удалось сопоставить с TIRADS.")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Исходное изображение")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image, cmap='gray')
    plt.contour(mask, colors='red', levels=[0.5])
    plt.legend(handles=[Line2D([0], [0], color='red', lw=2, label='Node')], loc='upper right')
    plt.title("Узел (маска)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = '../cropped_regions/thyroid_carotis_597102879_20250527_143045_20250527_143047.jpg'
    mask_path = '../sam_predictions/binary_masks/thyroid_carotis_597102879_20250527_143045_20250527_143047_binary_mask_0.png'

    process_custom_image(image_path, mask_path)