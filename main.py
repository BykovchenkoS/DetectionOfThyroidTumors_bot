import telebot
from telebot import types
from config import TOKEN

TOKEN = TOKEN
bot = telebot.TeleBot(TOKEN)


@bot.message_handler(commands=['start'])
def start(message):
    global cursor
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    hello_button = types.KeyboardButton('👋 Привет!')
    markup.add(hello_button)
    bot.send_message(message.chat.id, 'Здравствуйте!😊', reply_markup=markup)


@bot.message_handler(func=lambda message: True)
def handle_text(message):
    if "щитовидная" in message.text.lower():
        bot.reply_to(message, "⚠ Если у вас есть подозрения на опухоль, обратитесь к эндокринологу!")
    else:
        bot.reply_to(message, "Отправьте данные обследования для анализа.")


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    bot.reply_to(message, "📷 Изображение получено. Обработка... (здесь может быть нейросеть для анализа)")


if __name__ == '__main__':
    print("Бот запущен...")
    bot.polling()
