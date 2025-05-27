import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import os
from database import db


def determine_echogenicity(image_path, mask_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    masked_image = cv2.bitwise_and(image, image, mask=mask)

    mean_intensity = np.mean(masked_image[mask > 0])

    if mean_intensity > 180:
        return "Гиперэхогенный"
    elif mean_intensity > 120:
        return "Изоэхогенный"
    elif mean_intensity > 60:
        return "Гипоэхогенный"
    else:
        return "Анэхогенный"


def get_tirads_echogenicity_info(echogenicity_type):
    mapping = {
        "Анэхогенный": "Анэхогенный",
        "Гиперэхогенный": "Гиперэхогенный или изоэхогенный",
        "Изоэхогенный": "Гиперэхогенный или изоэхогенный",
        "Гипоэхогенный": "Гипоэхогенный",
        "Очень гипоэхогенный": "Очень гипоэхогенный"
    }

    db_option_name = mapping.get(echogenicity_type)
    if not db_option_name:
        return None

    query = """
        SELECT * FROM tirads_options 
        WHERE category = 'echogenicity' AND option_name = %s
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

    echogenicity_type = determine_echogenicity(image_path, mask_path)

    tirads_data = get_tirads_echogenicity_info(echogenicity_type)

    print("\nРезультат анализа эхогенности:")
    print(f"Тип эхогенности: {echogenicity_type}")
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
