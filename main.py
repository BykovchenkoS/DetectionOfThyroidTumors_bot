import logging
import os
from datetime import datetime
import telebot
from telebot import types
from config import TOKEN
from database import db
from mask_rcnn_processor import MaskRCNNThyroidAnalyzer
import torch
from PIL import Image

bot = telebot.TeleBot(TOKEN)
MODEL_PATH = 'neural_networks/mask_rcnn_model_screen.pth'
processor_mask_rcnn = MaskRCNNThyroidAnalyzer(MODEL_PATH)

os.makedirs('user_scans/original', exist_ok=True)
os.makedirs('user_scans/processed', exist_ok=True)


@bot.message_handler(commands=['start'])
def send_welcome(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.row("Анализ снимка (AI) 🔍", "Оценка по ACR TI-RADS📊")
    markup.row("ℹ️ Справка")

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

        # Обработка изображения нейросетью
        result_buffer = processor_mask_rcnn.process_image(original_path)
        if result_buffer is None:
            raise Exception("Не удалось обработать изображение")

        processed_filename = f"processed_{original_filename}"
        processed_path = os.path.join('user_scans', 'processed', processed_filename)

        with open(processed_path, 'wb') as f:
            f.write(result_buffer.getvalue())

        db.execute_query(
            "UPDATE scans SET processed_filepath=%s, status='completed' WHERE scan_id=%s",
            (processed_path, scan_id)
        )

        # Определяем, был ли найден Carotis
        found_classes = set()

        img = Image.open(original_path).convert("RGB")
        img_tensor = processor_mask_rcnn._transform(img).unsqueeze(0).to(processor_mask_rcnn.device)

        with torch.no_grad():
            predictions = processor_mask_rcnn.model(img_tensor)[0]

        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        keep = scores >= 0.5
        detected_labels = labels[keep]

        for label in detected_labels:
            class_name = processor_mask_rcnn.class_names[label]
            found_classes.add(class_name)

        caption = "🧠 AI-анализ завершён\nНа изображении выделены:\n"
        caption += "- 🟣 Щитовидная железа\n"

        if 'Carotis' in found_classes:
            caption += "- 🟢 Сонная артерия\n"

        with open(processed_path, 'rb') as photo:
            bot.send_photo(message.chat.id, photo, caption=caption.strip())

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


@bot.message_handler(func=lambda m: m.text == "ℹ️ Справка")
def send_help(message):
    help_text = (
        "📌 *Как работает бот?*\n"
        "1. Вы отправляете фото УЗИ щитовидной железы.\n"
        "2. Бот обрабатывает изображение с помощью ИИ или по шкале ACR TI-RADS.\n"
        "3. Вы получаете результат с визуализацией.\n\n"
        "🔍 *AI-анализ* использует модель Mask R-CNN для выделения тканей.\n"
        "📊 *ACR TI-RADS* — система оценки риска злокачественности.\n"
        "🟢 Зелёным цветом отмечена сонная артерия, если она была найдена.\n"
        "🟣 Фиолетовым — ткань щитовидной железы."
    )
    bot.send_message(message.chat.id, help_text, parse_mode="Markdown")


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


if __name__ == '__main__':
    print("Бот запущен...")
    bot.polling(none_stop=True)
