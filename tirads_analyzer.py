import cv2
import numpy as np
import os
from acr_ti_rads.echogenicity import determine_echogenicity
from acr_ti_rads.position import determine_node_position
from acr_ti_rads.borders import determine_border_type_by_pixels
from acr_ti_rads.inclusions import analyze_inclusions
from database import db
import logging


def save_tirads_result_to_db(scan_id, tirads_result):
    try:
        composition_option = db.fetch_one(
            "SELECT option_id FROM tirads_options WHERE category = 'composition' AND points = %s LIMIT 1",
            (tirads_result['details'].get('composition', 0),)
        )

        echogenicity_option = db.fetch_one(
            "SELECT option_id FROM tirads_options WHERE category = 'echogenicity' AND points = %s LIMIT 1",
            (tirads_result['details'].get('echogenicity', 0),)
        )

        position_option = db.fetch_one(
            "SELECT option_id FROM tirads_options WHERE category = 'position' AND points = %s LIMIT 1",
            (tirads_result['details'].get('position', 0),)
        )

        borders_option = db.fetch_one(
            "SELECT option_id FROM tirads_options WHERE category = 'borders' AND points = %s LIMIT 1",
            (tirads_result['details'].get('borders', 0),)
        )

        inclusions_option = db.fetch_one(
            "SELECT option_id FROM tirads_options WHERE category = 'inclusions' AND points = %s LIMIT 1",
            (tirads_result['details'].get('inclusions', 0),)
        )

        result = db.execute_query(
            """
            INSERT INTO tirads_results (
                scan_id,
                composition_id,
                echogenicity_id,
                position_id,
                borders_id,
                inclusions_id,
                total_score,
                category,
                malignancy_risk,
                description,
                recommendation
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                scan_id,
                composition_option['option_id'] if composition_option else None,
                echogenicity_option['option_id'] if echogenicity_option else None,
                position_option['option_id'] if position_option else None,
                borders_option['option_id'] if borders_option else None,
                inclusions_option['option_id'] if inclusions_option else None,
                tirads_result['total_score'],
                tirads_result['category'],
                tirads_result['risk'],
                tirads_result['description'],
                tirads_result['recommendation']
            )
        )
        logging.info(f"Результат ACR TI-RADS успешно сохранён в БД (result_id={result})")
        return result
    except Exception as e:
        logging.error(f"Не удалось сохранить результат ACR TI-RADS в БД: {e}")
        raise


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

    tirads_data = {
        "total_score": total_score,
        "features": features,
    }

    if total_score == 0:
        tirads_data.update({
            "category": "TR1",
            "risk": "Доброкачественно",
            "recommendation": "Без Тонкоигольной Аспирационной Биопсии (ТАБ)."
        })

    elif total_score == 1:
        tirads_data.update({
            "category": "TR2",
            "risk": "Не злокачественно",
            "recommendation": "Без Тонкоигольной Аспирационной Биопсии (ТАБ)."
        })

    elif total_score == 2:
        tirads_data.update({
            "category": "TR3",
            "risk": "Вероятность рака мала",
            "recommendation": "Тонкоигольная Аспирационная Биопсия (ТАБ) если размер ≥ 2.5 см.\nКонтроль при размере ≥ 1.5 см."
        })

    elif 3 <= total_score <= 6:
        tirads_data.update({
            "category": "TR4",
            "risk": "Вероятность рака умеренная",
            "recommendation": "Тонкоигольная Аспирационная Биопсия (ТАБ) если размер ≥ 1.5 см.\nКонтроль при размере ≥ 1 см."
        })

    else:
        tirads_data.update({
            "category": "TR5",
            "risk": "Вероятность рака значительная",
            "recommendation": "Тонкоигольная Аспирационная Биопсия (ТАБ) если размер ≥ 1 см.\nКонтроль при размере ≥ 0.5 см."
        })

    return tirads_data


def run_tirads_analysis(image_path, mask_path, composition_points=None):
    print(f"[DEBUG] Загрузка изображения: {image_path}")
    print(f"[DEBUG] Загрузка маски: {mask_path}")

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        raise ValueError("Не удалось загрузить изображение или маску.")

    composition_points = composition_points if composition_points is not None else 2

    echo_info = determine_echogenicity(image, mask)
    position_info = determine_node_position(mask)
    border_info = determine_border_type_by_pixels(image, mask)
    inclusion_info = analyze_inclusions(image, mask)

    features = {
        'composition': composition_points,
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

    try:
        analysis_result = run_tirads_analysis(image_path, mask_path)
        print("\nTI-RADS анализ завершён!\n")
        print(f"Категория: {analysis_result['category']}")
        print(f"Оценка риска: {analysis_result['risk']}")
        print("\nОписание узла:")
        print(analysis_result['description'])
        print("\nДетали оценки:")
        print(f"• Эхогенность: {analysis_result['details']['echogenicity']} балла")
        print(f"• Положение: {analysis_result['details']['position']} балла")
        print(f"• Границы: {analysis_result['details']['borders']} балла")
        print(f"• Включения: {analysis_result['details']['inclusions']} балла")
        print(f"---")
        print(f"Общий балл: {sum(analysis_result['details'].values())}")

        scan_id = 1
        result_id = save_tirads_result_to_db(scan_id, analysis_result)
        print(f"[INFO] Результат успешно записан в БД (result_id={result_id})")

    except Exception as e:
        print(f"[ERROR] Не удалось выполнить или сохранить анализ: {e}")
