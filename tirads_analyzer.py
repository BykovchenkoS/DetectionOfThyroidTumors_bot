import cv2
import numpy as np
import os
from acr_ti_rads.echogenicity import determine_echogenicity
from acr_ti_rads.position import determine_node_position
from acr_ti_rads.borders import determine_border_type_by_pixels
from acr_ti_rads.inclusions import analyze_inclusions


def get_node_description(features):
    descriptions = []

    echogenicity_map = {
        0: "анэхогенный",
        1: "гиперэхогенный или изоэхогенный",
        2: "гипоэхогенный"
    }
    echo_desc = echogenicity_map.get(features['echogenicity'], "неопределённой эхогенности")
    descriptions.append(f"Узел {echo_desc}.")

    position_map = {
        0: "шире, чем выше",
        1: "выше, чем шире"
    }
    pos_desc = position_map.get(features['position'], "неопределённой ориентации")
    descriptions.append(f"Имеет ориентацию '{pos_desc}'")

    borders_map = {
        0: "с ровными и чёткими границами",
        2: "с признаками выпячивания за пределы железы",
        3: "с выраженной инфильтрацией окружающих тканей"
    }
    border_desc = borders_map.get(features['borders'], "с неопределёнными границами")
    descriptions.append(f"{border_desc}.")

    inclusions_map = {
        0: "Патологических включений не выявлено",
        1: "Обнаружены мелкие гиперэхогенные включения",
        2: "Выявлены макрокальцинаты",
        3: "Обнаружены признаки периферического обызвествления"
    }
    incl_desc = inclusions_map.get(features['inclusions'], "Обнаружены неопределённые включения")
    descriptions.append(f"{incl_desc}.")

    full_description = " ".join(descriptions)

    total_score = sum(features.values())
    if total_score <= 1:
        full_description += " Морфологические характеристики соответствуют доброкачественному образованию."
    elif total_score <= 3:
        full_description += " Морфологические характеристики требуют динамического наблюдения."
    else:
        full_description += " Морфологические характеристики вызывают подозрение на злокачественный процесс."

    return full_description


def calculate_tirads_score(features):
    total_score = sum(features.values())

    if total_score == 0:
        return {
            "category": "TR1",
            "risk": "Доброкачественно",
            "recommendation": "Без Тонкоигольной Аспирационной Биопсии (ТАБ)."
        }
    elif total_score == 1:
        return {
            "category": "TR2",
            "risk": "Не злокачественно",
            "recommendation": "Без Тонкоигольной Аспирационной Биопсии (ТАБ)."
        }
    elif total_score == 2:
        return {
            "category": "TR3",
            "risk": "Вероятность рака мала",
            "recommendation": "Тонкоигольная Аспирационная Биопсия (ТАБ) если размер ≥ 2.5 см.\nКонтроль при размере ≥ 1.5 см."
        }
    elif total_score >= 3 and total_score <= 6:
        return {
            "category": "TR4",
            "risk": "Вероятность рака умеренная",
            "recommendation": "Тонкоигольная Аспирационная Биопсия (ТАБ) если размер ≥ 1.5 см.\nКонтроль при размере ≥ 1 см."
        }
    else:
        return {
            "category": "TR5",
            "risk": "Вероятность рака значительная",
            "recommendation": "Тонкоигольная Аспирационная Биопсия (ТАБ) если размер ≥ 1 см.\nКонтроль при размере ≥ 0.5 см."
        }


def run_tirads_analysis(image_path, mask_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        raise ValueError("Не удалось загрузить изображение или маску.")

    echo_info = determine_echogenicity(image, mask)
    position_info = determine_node_position(mask)
    border_info = determine_border_type_by_pixels(image, mask)
    inclusion_info = analyze_inclusions(image, mask)

    features = {
        'echogenicity': echo_info.get('points', 0),
        'position': position_info.get('points', 0),
        'borders': border_info.get('points', 0),
        'inclusions': inclusion_info.get('points', 0)
    }

    result = calculate_tirads_score(features)
    result['details'] = features
    result['description'] = get_node_description(features)

    return result


if __name__ == "__main__":
    image_path = 'cropped_regions/thyroid_carotis_597102879_20250527_143045_20250527_143047.jpg'
    mask_path = 'sam_predictions/binary_masks/thyroid_carotis_597102879_20250527_143045_20250527_143047_binary_mask_0.png'

    analysis_result = run_tirads_analysis(image_path, mask_path)
    print("\nTI-RADS анализ завершён!\n")
    print(f"Категория: {analysis_result['category']}")
    print(f"Оценка риска: {analysis_result['risk']}")
    print(f"Рекомендация: {analysis_result['recommendation']}")
    print("\nОписание узла:")
    print(analysis_result['description'])
    print("\nДетали оценки:")
    print(f"• Эхогенность: {analysis_result['details']['echogenicity']} балла")
    print(f"• Положение: {analysis_result['details']['position']} балла")
    print(f"• Границы: {analysis_result['details']['borders']} балла")
    print(f"• Включения: {analysis_result['details']['inclusions']} балла")
    print(f"---")
    print(f"Общий балл: {sum(analysis_result['details'].values())}")
