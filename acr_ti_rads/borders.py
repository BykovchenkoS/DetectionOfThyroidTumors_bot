import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import os
from database import db


def extract_boundary_pixels(mask):
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    boundary = mask - eroded
    return boundary


def analyze_boundary_intensity(image, boundary):
    boundary_pixels = image[boundary > 0]
    mean_intensity = np.mean(boundary_pixels)
    std_intensity = np.std(boundary_pixels)
    return mean_intensity, std_intensity


def analyze_boundary_texture(image, boundary):
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    boundary_gradient = gradient_magnitude[boundary > 0]
    mean_gradient = np.mean(boundary_gradient)
    return mean_gradient


def analyze_contour_shape(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return float('inf')
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    compactness = (perimeter**2) / (4 * np.pi * area)
    return compactness


def determine_border_type_by_pixels(image, mask):
    boundary = extract_boundary_pixels(mask)

    mean_intensity, std_intensity = analyze_boundary_intensity(image, boundary)
    mean_gradient = analyze_boundary_texture(image, boundary)
    compactness = analyze_contour_shape(mask)

    print(f"std_intensity: {std_intensity}, mean_gradient: {mean_gradient}, compactness: {compactness}")

    if std_intensity < 30 and mean_gradient < 40 and compactness < 1.6:
        return "Неявная"
    elif std_intensity < 35 and mean_gradient < 80 and compactness < 1.8:
        return "Явная"
    elif std_intensity < 45 and mean_gradient < 100 and compactness < 2.0:
        return "Волнообразная (бугристая)"
    else:
        return "Выпячивание из железы"


def get_tirads_border_info(border_type):
    mapping = {
        "Неявная": "Ровные",
        "Явная": "Нечеткие",
        "Волнообразная (бугристая)": "Дольчатые (бугристые)",
        "Выпячивание из железы": "Выход за пределы железы"
    }

    db_option_name = mapping.get(border_type)
    if not db_option_name:
        return None

    query = """
        SELECT * FROM tirads_options WHERE category = 'borders' AND option_name = %s
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

    border_type = determine_border_type_by_pixels(image, mask)

    tirads_data = get_tirads_border_info(border_type)

    print("\nРезультат анализа")
    print(f"Тип границы узла: {border_type}")
    if tirads_data:
        print(f"Соответствует TIRADS опции: {tirads_data['option_name']}")
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