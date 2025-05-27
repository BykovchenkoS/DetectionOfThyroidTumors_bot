import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import os
from database import db


def determine_node_position(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "Контур узла не найден"
    contour = max(contours, key=cv2.contourArea)

    x, y, width, height = cv2.boundingRect(contour)
    aspect_ratio = width / height

    if aspect_ratio > 1:
        return "Продольно доли (шире, чем выше)"
    else:
        return "Поперечно доли (выше, чем шире)"


def get_tirads_position_info(position_type):
    mapping = {
        "Продольно доли (шире, чем выше)": "Шире, чем выше",
        "Поперечно доли (выше, чем шире)": "Выше, чем шире"
    }

    db_option_name = mapping.get(position_type)
    if not db_option_name:
        return None

    query = """
        SELECT * FROM tirads_options WHERE category = 'position' AND option_name = %s
    """
    result = db.fetch_one(query, (db_option_name,))
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

    position_type = determine_node_position(mask)
    tirads_data = get_tirads_position_info(position_type)

    print("\nРезультат анализа положения:")
    print(f"Тип положения: {position_type}")
    if tirads_data:
        print(f"Соответствует TIRADS опции: {tirads_data['option_name']}")
        print(f"Баллы: {tirads_data['points']}")
        print(f"Описание: {tirads_data['description']}")
    else:
        print("Не удалось сопоставить с TIRADS.")

    # Визуализация
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
