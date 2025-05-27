import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import os
from database import db


def determine_node_position(mask):
    if mask is None:
        raise ValueError("Маска не была загружена корректно")

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"type": "Контур не найден", "points": 0}

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h

    tirads_type = "Шире, чем выше" if aspect_ratio > 1 else "Выше, чем шире"

    query = """
        SELECT * FROM tirads_options 
        WHERE category = 'position' AND option_name = %s
    """
    result = db.fetch_one(query, (tirads_type,))
    return {
        "type": tirads_type,
        "points": result["points"] if result else 0
    }


def get_tirads_position_info(position_type):
    query = """
        SELECT * FROM tirads_options WHERE category = 'position' AND option_name = %s
    """
    result = db.fetch_one(query, (position_type,))
    return result


def process_custom_image(image_path, mask_path):
    if not os.path.exists(image_path):
        print(f"Файл изображения не найден: {image_path}")
        return
    if not os.path.exists(mask_path):
        print(f"Файл маски не найден: {mask_path}")
        return

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print("Ошибка: Не удалось загрузить изображение или маску.")
        return

    position_result = determine_node_position(mask)
    position_type = position_result["type"]

    tirads_data = get_tirads_position_info(position_type)

    print("\nРезультат анализа положения:")
    print(f"Тип положения: {position_type}")

    if tirads_data:
        print(f"\nСоответствует TIRADS опции: {tirads_data['option_name']}")
        print(f"Баллы: {tirads_data['points']}")
        print(f"Описание: {tirads_data['description']}")
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
