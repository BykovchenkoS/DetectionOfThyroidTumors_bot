import logging
import os
from datetime import datetime
import telebot
from PIL.ImageDraw import ImageDraw
from telebot import types
from config import TOKEN
from database import db
from mask_rcnn_processor import MaskRCNNThyroidAnalyzer
from yolo_sam_processor import YOLOSAMNodeAnalyzer
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
import time
from threading import Timer


bot = telebot.TeleBot(TOKEN)
MODEL_PATH = 'neural_networks/mask_rcnn_model_screen.pth'
processor_mask_rcnn = MaskRCNNThyroidAnalyzer(MODEL_PATH)

os.makedirs('user_scans/original', exist_ok=True)
os.makedirs('user_scans/processed', exist_ok=True)


@bot.message_handler(commands=['start'])
def send_welcome(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.row("Анализ снимка (AI) 🔍", "Оценка по ACR TI-RADS📊")

    try:
        db.execute_query(
            "INSERT INTO users (user_id, username) VALUES (%s, %s) ON DUPLICATE KEY UPDATE username=%s",
            (message.from_user.id, message.from_user.username, message.from_user.username)
        )
    except Exception as e:
        logging.error(f"Error saving user: {e}")

    bot.send_message(
        message.chat.id,
        "👋 Добро пожаловать! Я помогу проанализировать снимок щитовидной железы.\n"
        "Выберите тип анализа:",
        reply_markup=markup
    )


@bot.message_handler(func=lambda m: m.text == "Анализ снимка (AI) 🔍")
def request_ai_scan(message):
    bot.send_message(message.chat.id,
                     "📤 Отправьте фото УЗИ щитовидной железы для AI-анализа. Убедитесь, что снимок четкий и захватывает всю область.")


@bot.message_handler(func=lambda m: m.text == "Оценка по ACR TI-RADS📊")
def request_tirads_scan(message):
    bot.send_message(message.chat.id,
                     "📤 Отправьте фото УЗИ щитовидной железы для анализа по шкале ACR TI-RADS. Убедитесь, что снимок четкий и захватывает всю область.")


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        file_ext = file_info.file_path.split('.')[-1]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_filename = f"{message.from_user.id}_{timestamp}.{file_ext}"
        original_path = os.path.join('user_scans', 'original', original_filename)

        with open(original_path, 'wb') as new_file:
            new_file.write(downloaded_file)

        analysis_type = 'ai'
        if message.reply_to_message and message.reply_to_message.text.startswith("Отправьте фото УЗИ"):
            analysis_type = 'ai' if "AI-анализа" in message.reply_to_message.text else 'ti-rads'

        scan_id = db.execute_query(
            "INSERT INTO scans (user_id, original_filepath, analysis_type) VALUES (%s, %s, %s)",
            (message.from_user.id, original_path, analysis_type)
        )

        # Обработка изображения Mask R-CNN
        result_buffer, prediction_dict, combined_cropped_path = processor_mask_rcnn.process_image(original_path)

        if result_buffer is None or prediction_dict is None:
            raise Exception("Не удалось обработать изображение")

        img = Image.open(original_path).convert("RGB")

        processed_filename = f"processed_{original_filename}"
        processed_path = os.path.join('user_scans', 'processed', processed_filename)

        with open(processed_path, 'wb') as f:
            f.write(result_buffer.getvalue())

        db.execute_query(
            "UPDATE scans SET processed_filepath=%s, status='completed' WHERE scan_id=%s",
            (processed_path, scan_id)
        )

        # Определяем, был ли найден Carotis
        labels = prediction_dict['labels'].cpu().numpy()
        scores = prediction_dict['scores'].cpu().numpy()
        keep = scores >= 0.5
        detected_labels = labels[keep]

        found_classes = set()
        for label in detected_labels:
            class_name = processor_mask_rcnn.class_names[label]
            found_classes.add(class_name)

        # Обрезаем изображение по объединённой области Thyroid + Carotis
        combined_cropped_path = processor_mask_rcnn._crop_combined_thyroid_carotis(img, prediction_dict, original_path)

        caption = "Анализирую... 🧠\n\nНа изображении выделены:\n"
        caption += "🟣 Щитовидная железа\n"
        if 'Carotis' in found_classes:
            caption += "🟢 Сонная артерия\n"

        messages_to_delete = []

        with open(processed_path, 'rb') as photo:
            sent_msg = bot.send_photo(message.chat.id, photo, caption=caption.strip())
            messages_to_delete.append(sent_msg.message_id)

        # Если есть обрезанное изображение, сохраняем его для дальнейшей работы
        if combined_cropped_path:
            print(f"[DEBUG] Обрезанное изображение сохранено: {combined_cropped_path}")

            # Детекция узла через YOLO + SAM
            yolo_sam_processor = YOLOSAMNodeAnalyzer(
                yolo_weights_path='neural_networks/train8_node_yolo12/weights/best.pt',
                sam_checkpoint_path='neural_networks/sam_vit_h_4b8939.pth',
                sam_finetuned_path='neural_networks/sam_best_node.pth'
            )

            masks, mask_vis_path = yolo_sam_processor.process(combined_cropped_path)

            if masks and mask_vis_path:
                with open(mask_vis_path, 'rb') as mask_file:
                    sent_msg = bot.send_photo(message.chat.id, mask_file, caption="🔴 Детекция узла завершена")
                    messages_to_delete.append(sent_msg.message_id)
                collage = create_collage(processed_path, mask_vis_path)
                caption = "✅ AI-анализ выполнен"
            else:
                collage = create_single_image_collage(processed_path)
                caption = "✅ AI-анализ выполнен"

            if collage:
                collage_path = os.path.join('user_scans', 'processed', f"collage_{timestamp}.png")
                try:
                    collage.save(collage_path)
                    with open(collage_path, 'rb') as collage_file:
                        sent_msg = bot.send_photo(
                            message.chat.id,
                            collage_file,
                            caption=caption
                        )

                    # Запускаем таймер для удаления предыдущих сообщений через 5 секунд
                    Timer(5.0, delete_messages, args=[message.chat.id, messages_to_delete]).start()

                except Exception as e:
                    logging.error(f"Error saving/sending collage: {e}")
                    bot.send_message(message.chat.id, "⚠ Не удалось создать коллаж")
            else:
                bot.send_message(message.chat.id, "⚠ Не удалось создать коллаж")

        # Оценка анализа
        markup_rate = types.InlineKeyboardMarkup(row_width=5)
        markup_rate.add(
            types.InlineKeyboardButton("1", callback_data=f"rate_{scan_id}_1"),
            types.InlineKeyboardButton("2", callback_data=f"rate_{scan_id}_2"),
            types.InlineKeyboardButton("3", callback_data=f"rate_{scan_id}_3"),
            types.InlineKeyboardButton("4", callback_data=f"rate_{scan_id}_4"),
            types.InlineKeyboardButton("5", callback_data=f"rate_{scan_id}_5")
        )
        bot.send_message(
            message.chat.id,
            "⭐️ Оцените качество анализа ⭐️",
            reply_markup=markup_rate
        )

    except Exception as e:
        logging.error(f"Error processing photo: {e}")
        bot.reply_to(message, "⚠ Произошла ошибка при обработке изображения. Пожалуйста, попробуйте позже.")


def delete_messages(chat_id, message_ids):
    try:
        for msg_id in message_ids:
            try:
                bot.delete_message(chat_id, msg_id)
            except Exception as e:
                logging.error(f"Error deleting message {msg_id}: {e}")
    except Exception as e:
        logging.error(f"Error in delete_messages: {e}")


@bot.callback_query_handler(func=lambda call: call.data.startswith('rate_'))
def handle_rating(call):
    try:
        _, scan_id, rating = call.data.split('_')
        rating = int(rating)
        if not 1 <= rating <= 5:
            raise ValueError("Недопустимая оценка")

        scan = db.fetch_one("SELECT analysis_type FROM scans WHERE scan_id = %s", (scan_id,))
        if not scan:
            raise Exception("Scan not found")
        analysis_type = scan['analysis_type']

        db.execute_query(
            "UPDATE scans SET user_rating=%s WHERE scan_id=%s",
            (rating, scan_id)
        )

        bot.edit_message_text(
            "Спасибо за оценку! Ваше мнение поможет улучшить сервис ☺️",
            call.message.chat.id,
            call.message.message_id
        )

        if analysis_type == 'ai':
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
            markup.add(types.KeyboardButton("Да"), types.KeyboardButton("Нет"))
            msg = bot.send_message(
                call.message.chat.id,
                "🔁 Хотите также провести оценку этого снимка по шкале ACR TI-RADS?",
                reply_markup=markup
            )
            bot.register_next_step_handler(msg, lambda m: handle_tirads_after_ai(m, scan_id))

    except ValueError as e:
        logging.error(f"Invalid rating value: {e}")
        bot.answer_callback_query(call.id, "Ошибка обработки оценки")
    except Exception as e:
        logging.error(f"Error saving rating: {e}")
        bot.answer_callback_query(call.id, "⚠ Ошибка сохранения, попробуйте позже.")


def handle_tirads_after_ai(message, scan_id):
    if message.text.lower() == 'да':
        bot.send_message(message.chat.id, "🔄 Выполняется ACR TI-RADS анализ...")
        scan = db.fetch_one("SELECT original_filepath, user_id FROM scans WHERE scan_id = %s", (scan_id,))
        if not scan:
            bot.send_message(message.chat.id, "❌ Не удалось найти исходный снимок.")
            return

        original_path = scan["original_filepath"]
        user_id = scan["user_id"]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        processed_filename = f"tirads_processed_{user_id}_{timestamp}.jpg"
        processed_path = os.path.join('user_scans', 'processed', processed_filename)

        with open(original_path, 'rb') as orig, open(processed_path, 'wb') as proc:
            proc.write(orig.read())

        new_scan_id = db.execute_query(
            "INSERT INTO scans (user_id, original_filepath, processed_filepath, analysis_type, status) "
            "VALUES (%s, %s, %s, %s, 'completed')",
            (user_id, original_path, processed_path, 'tirads')
        )

        result_text = (
            "✅ Анализ по шкале ACR TI-RADS:\n"
            "Категория 4A\n"
            "Риск злокачественности ~5-10%\n\n"
            "Рекомендация: провести тонкоигольную аспирационную биопсию."
        )
        bot.send_message(message.chat.id, result_text)

        markup_rate = types.InlineKeyboardMarkup(row_width=5)
        markup_rate.add(
            types.InlineKeyboardButton("1", callback_data=f"rate_{new_scan_id}_1"),
            types.InlineKeyboardButton("2", callback_data=f"rate_{new_scan_id}_2"),
            types.InlineKeyboardButton("3", callback_data=f"rate_{new_scan_id}_3"),
            types.InlineKeyboardButton("4", callback_data=f"rate_{new_scan_id}_4"),
            types.InlineKeyboardButton("5", callback_data=f"rate_{new_scan_id}_5")
        )
        bot.send_message(
            message.chat.id,
            "⭐️ Оцените качество анализа ⭐️",
            reply_markup=markup_rate
        )

    else:
        bot.send_message(message.chat.id, "👌 Хорошо, пропускаем дополнительный анализ.")


def create_collage(original_image_path, cropped_image_with_nodule_path):
    try:
        original_img = Image.open(original_image_path)
        nodule_img = Image.open(cropped_image_with_nodule_path)

        border_size = 40
        padding_color = (255, 255, 255)
        text_padding = 30
        title_height = 80
        caption_height = 80

        scale_factor = 0.8
        nodule_img = nodule_img.resize(
            (int(nodule_img.width * scale_factor), int(nodule_img.height * scale_factor)),
            Image.Resampling.LANCZOS
        )

        max_img_height = max(original_img.height, nodule_img.height) + border_size * 2

        original_with_border = ImageOps.expand(original_img, border=border_size, fill=padding_color)
        nodule_with_border = ImageOps.expand(nodule_img, border=border_size, fill=padding_color)

        def add_vertical_padding(img, target_height):
            current_height = img.height
            if current_height < target_height:
                top_pad = (target_height - current_height) // 2
                bottom_pad = target_height - current_height - top_pad
                return ImageOps.expand(img, border=(0, top_pad, 0, bottom_pad), fill=padding_color)
            return img

        original_with_border = add_vertical_padding(original_with_border, max_img_height)
        nodule_with_border = add_vertical_padding(nodule_with_border, max_img_height)

        collage_width = original_with_border.width + nodule_with_border.width
        collage_height = title_height + max_img_height + caption_height
        collage = Image.new('RGB', (collage_width, collage_height), color=padding_color)

        collage.paste(original_with_border, (0, title_height))
        collage.paste(nodule_with_border, (original_with_border.width, title_height))

        draw = ImageDraw.Draw(collage)

        font = ImageFont.truetype("arial.ttf", 24)
        font_large = ImageFont.truetype("arial.ttf", 28)

        title = "Результаты анализа УЗИ щитовидной железы"
        title_width = draw.textlength(title, font=font_large)
        draw.text(
            (collage.width // 2 - title_width // 2, title_height // 2 - 15),
            title, fill="black", font=font_large
        )

        left_text_line1 = "Обнаруженные щитовидная железа"
        left_text_line2 = "и сонная артерия"

        left_text_width1 = draw.textlength(left_text_line1, font=font)
        left_text_width2 = draw.textlength(left_text_line2, font=font)

        draw.text(
            (original_with_border.width // 2 - left_text_width1 // 2,
             title_height + max_img_height + 10),
            left_text_line1, fill="black", font=font
        )
        draw.text(
            (original_with_border.width // 2 - left_text_width2 // 2,
             title_height + max_img_height + 40),
            left_text_line2, fill="black", font=font
        )

        right_text = "Обнаруженный узел"
        right_text_width = draw.textlength(right_text, font=font)
        draw.text(
            (original_with_border.width + nodule_with_border.width // 2 - right_text_width // 2,
             title_height + max_img_height + 25),
            right_text, fill="black", font=font
        )

        return collage

    except Exception as e:
        logging.error(f"Error creating collage: {e}")
        return None


def create_single_image_collage(original_image_path):
    try:
        original_img = Image.open(original_image_path)
        border_size = 40
        padding_color = (255, 255, 255)
        text_padding = 30
        title_height = 100
        caption_height = 120

        img_with_border = ImageOps.expand(original_img, border=border_size, fill=padding_color)

        collage_width = img_with_border.width + 2 * border_size
        collage_height = title_height + img_with_border.height + caption_height
        collage = Image.new('RGB', (collage_width, collage_height), color=padding_color)

        collage.paste(img_with_border, (border_size, title_height))

        draw = ImageDraw.Draw(collage)

        font = ImageFont.truetype("arial.ttf", 24)
        font_large = ImageFont.truetype("arial.ttf", 28)

        title = "Результаты анализа УЗИ щитовидной железы"
        title_width = draw.textlength(title, font=font_large)
        draw.text(
            (collage.width // 2 - title_width // 2, title_height // 2 - 15),
            title, fill="black", font=font_large
        )

        text_line1 = "Обнаруженные щитовидная железа"
        text_line2 = "и сонная артерия"
        text_line3 = "Узлы не обнаружены"

        text_y_start = title_height + img_with_border.height + 10
        line_spacing = 30

        text_width = draw.textlength(text_line1, font=font)
        draw.text(
            (collage.width // 2 - text_width // 2, text_y_start),
            text_line1, fill="black", font=font
        )

        text_width = draw.textlength(text_line2, font=font)
        draw.text(
            (collage.width // 2 - text_width // 2, text_y_start + line_spacing),
            text_line2, fill="black", font=font
        )

        text_width = draw.textlength(text_line3, font=font)
        draw.text(
            (collage.width // 2 - text_width // 2, text_y_start + 2 * line_spacing),
            text_line3, fill="red", font=font
        )

        return collage

    except Exception as e:
        logging.error(f"Error creating single image collage: {e}")
        return None


if __name__ == '__main__':
    bot.delete_webhook()
    print("Бот запущен...")
    bot.polling(none_stop=True)
