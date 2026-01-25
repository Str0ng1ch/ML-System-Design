import json
import logging
import os
import re
from datetime import datetime
from typing import List

import requests
from telegram import ReplyKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    Defaults,
    MessageHandler,
    filters,
)

from backend.prompt_security import prompt_security

# Настройка логирования
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Состояния разговора
MAIN_MENU, WAITING_ASSIGNMENT, WAITING_ESSAY, WAITING_RATING, WAITING_COMMENT = range(5)
BOT_TOKEN = os.getenv("BOT_TOKEN")
BACKEND_URL = "http://localhost:5000"
USER_DATA = {}

# Константы для кнопок
BTN_UPLOAD_ASSIGNMENT = "Шаг 1. Загрузить задание от преподавателя"
BTN_GET_RECOMMENDATIONS = "Шаг 2. Получить рекомендации"
BTN_RATE_BOT = "Шаг 3. Оценить работу бота"
BTN_CANCEL = "Отменить"
BTN_SKIP = "Пропустить"

# Пути к данным
DATA_DIR = "/app/data"
USAGE_DIR = os.path.join(DATA_DIR, "usage")


def split_text_for_telegram(text: str, max_len: int = 4096) -> List[str]:
    """
    Разбивает текст на части для отправки в Telegram.
    Args:
        text (str): Текст для разделения
        max_len (int): Максимальная длина части текста
    Returns:
        List[str]: Список частей текста
    """
    parts = []
    remaining = text or ""
    while remaining:
        if len(remaining) <= max_len:
            parts.append(remaining)
            break
        window = remaining[:max_len]
        # 1) Пытаемся разорвать по границе абзацев (\n\n)
        para_pos = window.rfind("\n\n")
        if para_pos != -1:
            split_at = para_pos
            chunk = remaining[:split_at].rstrip()
            # пропускаем все последующие переводы строк
            j = split_at
            while j < len(remaining) and remaining[j] == "\n":
                j += 1
            remaining = remaining[j:]
            if chunk:
                parts.append(chunk)
            continue
        # 2) Иначе разрываем по последнему пробелу/переводу строки/табуляции
        last_ws = max(window.rfind("\n"), window.rfind(" "), window.rfind("\t"))
        if last_ws <= 0:
            last_ws = max_len
        chunk = remaining[:last_ws].rstrip()
        remaining = remaining[last_ws:].lstrip()
        if chunk:
            parts.append(chunk)
    return parts


def _today_str() -> str:
    """
    Возвращает текущую дату в формате строки.
    Returns:
        str: Текущая дата в формате YYYY-MM-DD
    """
    return datetime.now().strftime("%Y-%m-%d")


def _usage_file_path(user_id: int) -> str:
    """
    Возвращает путь к файлу использования для пользователя.
    Args:
        user_id (int): ID пользователя
    Returns:
        str: Путь к файлу использования
    """

    os.makedirs(USAGE_DIR, exist_ok=True)
    return os.path.join(USAGE_DIR, f"{user_id}.json")


def has_daily_quota(user_id: int) -> bool:
    """
    Проверяет, есть ли у пользователя дневной лимит на использование.
    Args:
        user_id (int): ID пользователя
    Returns:
        bool: True если лимит не исчерпан, иначе False
    """
    path = _usage_file_path(user_id)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("date") == _today_str() and int(data.get("count", 0)) >= 1:
            return False
        return True
    except Exception:
        return True


def record_daily_use(user_id: int) -> None:
    """
    Записывает использование бота пользователем.
    Args:
        user_id (int): ID пользователя
    """
    path = _usage_file_path(user_id)
    try:
        payload = {"date": _today_str(), "count": 1}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to persist usage for user {user_id}: {e}")


def get_main_menu_keyboard():
    """
    Создает клавиатуру главного меню.
    Returns:
        List[List[str]]: Клавиатура главного меню
    """
    return [[BTN_UPLOAD_ASSIGNMENT, BTN_GET_RECOMMENDATIONS], [BTN_RATE_BOT]]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обрабатывает команду /start.
    Args:
        update (Update): Объект обновления Telegram
        context (ContextTypes.DEFAULT_TYPE): Контекст бота
    Returns:
        int: Следующее состояние разговора
    """
    user = update.message.from_user
    USER_DATA[user.id] = {"username": user.username, "first_name": user.first_name}

    keyboard = get_main_menu_keyboard()
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)

    welcome_text = """
Приветствую, дорогой урбанист!

Я - Джейн, ваш AI-ассистент в мире городских исследований, который поможет тебе улучшить свои работы,
предоставляя конструктивные рекомендации на основе академических источников.

Чтобы получить рекомендацию сделай 3 простых шага:
1) Загрузи задание от преподавателя (.docx или .txt)
2) Загрузи свою работу (.docx или .txt) и получи рекомендацию
3) Оцени работу бота, чтобы помочь нам стать лучше! :)

Внимание: Доступно не более одной рекомендации в день!
Если бот не отвечает или ведёт себя странно, отправьте команду /start, чтобы перезагрузить диалог.
"""

    await update.message.reply_text(welcome_text, reply_markup=reply_markup)
    return MAIN_MENU


async def handle_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обрабатывает выбор в главном меню.
    Args:
        update (Update): Объект обновления Telegram
        context (ContextTypes.DEFAULT_TYPE): Контекст бота
    Returns:
        int: Следующее состояние разговора
    """
    text = update.message.text
    user_id = update.message.from_user.id

    if text == BTN_UPLOAD_ASSIGNMENT:
        cancel_keyboard = [[BTN_CANCEL]]
        reply_markup = ReplyKeyboardMarkup(cancel_keyboard, resize_keyboard=True, one_time_keyboard=False)
        await update.message.reply_text(
            "Загрузите файл задания (.docx или .txt). Он сохранится один раз и будет использован для всех проверок.",
            reply_markup=reply_markup,
        )
        return WAITING_ASSIGNMENT

    elif text == BTN_GET_RECOMMENDATIONS:
        # Проверяем, что задание уже загружено
        try:
            resp = requests.get(
                f"{BACKEND_URL}/assignment/status",
                params={"user_id": str(user_id)},
                timeout=10,
            )
            has = bool(resp.json().get("has_assignment")) if resp.status_code == 200 else False
        except Exception as e:
            logger.error(f"Error checking assignment status: {e}")
            has = False

        if not has:
            keyboard = get_main_menu_keyboard()
            reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)
            await update.message.reply_text(
                "Сначала загрузите задание: нажмите «Загрузить задание от преподавателя» и "
                "отправьте файл (.docx или .txt).",
                reply_markup=reply_markup,
            )
            return MAIN_MENU

        # Лимит: одна рекомендация в день
        if not has_daily_quota(user_id):
            keyboard = get_main_menu_keyboard()
            reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)
            await update.message.reply_text(
                "Лимит: не более одной рекомендации в день. Попробуйте завтра.",
                reply_markup=reply_markup,
            )
            return MAIN_MENU

        cancel_keyboard = [[BTN_CANCEL]]
        reply_markup = ReplyKeyboardMarkup(cancel_keyboard, resize_keyboard=True, one_time_keyboard=False)
        await update.message.reply_text(
            "Отправьте файл вашей работы (.docx или .txt). Я проанализирую его с учётом сохранённого задания. "
            "Будьте внимательны: установлен лимит на одну рекомендацию в день!",
            reply_markup=reply_markup,
        )
        return WAITING_ESSAY
    elif text == BTN_RATE_BOT:
        rating_keyboard = [["1", "2", "3", "4", "5"], [BTN_CANCEL]]
        reply_markup = ReplyKeyboardMarkup(rating_keyboard, resize_keyboard=True, one_time_keyboard=False)

        await update.message.reply_text(
            "Пожалуйста, оцените мою работу по шкале от 1 до 5, где 5 - отлично, а 1 - плохо:",
            reply_markup=reply_markup,
        )
        return WAITING_RATING
    else:
        # Обработка неправильных действий в главном меню
        keyboard = get_main_menu_keyboard()
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)

        await update.message.reply_text(
            "Пожалуйста, выберите одну из предложенных опций:",
            reply_markup=reply_markup,
        )
        return MAIN_MENU


async def handle_incorrect_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обрабатывает неправильные действия (например, загрузка файла не в том состоянии).
    Args:
        update (Update): Объект обновления Telegram
        context (ContextTypes.DEFAULT_TYPE): Контекст бота
    Returns:
        int: Текущее или следующее состояние разговора
    """
    user_id = update.message.from_user.id
    current_state = context.user_data.get("state", MAIN_MENU)

    if current_state == MAIN_MENU:
        keyboard = get_main_menu_keyboard()
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)

        await update.message.reply_text(
            "Пожалуйста, сначала выберите опцию 'Получить рекомендации', а затем загрузите файл.",
            reply_markup=reply_markup,
        )
        return MAIN_MENU
    else:
        return current_state


async def handle_rating(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обрабатывает оценку от пользователя.
    Args:
        update (Update): Объект обновления Telegram
        context (ContextTypes.DEFAULT_TYPE): Контекст бота
    Returns:
        int: Следующее состояние разговора
    """
    rating = update.message.text
    user_id = update.message.from_user.id

    if rating == "Отменить":
        return await cancel(update, context)

    if rating not in ["1", "2", "3", "4", "5"]:
        rating_keyboard = [["1", "2", "3", "4", "5"], ["Отменить"]]
        reply_markup = ReplyKeyboardMarkup(rating_keyboard, resize_keyboard=True, one_time_keyboard=False)
        await update.message.reply_text("Пожалуйста, выберите оценку от 1 до 5:", reply_markup=reply_markup)
        return WAITING_RATING

    if user_id not in USER_DATA:
        USER_DATA[user_id] = {}
    USER_DATA[user_id]["rating"] = rating

    await update.message.reply_text(
        "Спасибо за оценку! Теперь, пожалуйста, напишите ваш комментарий или нажмите 'Пропустить':",
        reply_markup=ReplyKeyboardMarkup([["Пропустить"]], resize_keyboard=True, one_time_keyboard=False),
    )
    return WAITING_COMMENT


async def handle_comment(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обрабатывает комментарий от пользователя.
    Args:
        update (Update): Объект обновления Telegram
        context (ContextTypes.DEFAULT_TYPE): Контекст бота
    Returns:
        int: Следующее состояние разговора
    """
    comment = update.message.text
    user_id = update.message.from_user.id

    if comment == "Пропустить":
        comment = ""

    if user_id not in USER_DATA:
        USER_DATA[user_id] = {}
    USER_DATA[user_id]["comment"] = comment

    try:
        payload = {
            "user_id": user_id,
            "rating": USER_DATA[user_id].get("rating", ""),
            "comment": comment,
        }

        response = requests.post(f"{BACKEND_URL}/feedback", json=payload, timeout=10)

        if response.status_code == 200:
            await update.message.reply_text("Спасибо за ваш отзыв! Он поможет мне стать лучше.")
        else:
            await update.message.reply_text("Спасибо за отзыв! (При сохранении произошла ошибка, но мы его учтем)")
    except Exception as e:
        logger.error(f"Error sending feedback: {e}")
        await update.message.reply_text("Спасибо за отзыв! (При сохранении произошла ошибка, но мы его учтем)")

    keyboard = get_main_menu_keyboard()
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)

    await update.message.reply_text("Чем еще могу помочь?", reply_markup=reply_markup)
    return MAIN_MENU


async def handle_assignment_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обрабатывает загрузку файла задания.
    Args:
        update (Update): Объект обновления Telegram
        context (ContextTypes.DEFAULT_TYPE): Контекст бота
    Returns:
        int: Следующее состояние разговора
    """
    user_id = update.message.from_user.id

    if not update.message.document or (
        not update.message.document.file_name.endswith(".txt")
        and not update.message.document.file_name.endswith(".docx")
    ):
        await update.message.reply_text("Неверный формат файла. Поддерживаются только .txt и .docx.")
        return WAITING_ASSIGNMENT

    try:
        file = await update.message.document.get_file()
        file_bytes = await file.download_as_bytearray()

        files = {"file": (update.message.document.file_name, file_bytes)}
        data = {"user_id": str(user_id)}

        await update.message.reply_text("Сохраняю задание...")
        response = requests.post(f"{BACKEND_URL}/assignment", files=files, data=data, timeout=30)

        if response.status_code == 200:
            keyboard = get_main_menu_keyboard()
            reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)
            await update.message.reply_text(
                "Задание сохранено. Теперь можно загружать работы для проверки.",
                reply_markup=reply_markup,
            )
            return MAIN_MENU
        else:
            await update.message.reply_text("Ошибка при сохранении задания. Попробуйте ещё раз.")
            return WAITING_ASSIGNMENT
    except requests.exceptions.Timeout:
        await update.message.reply_text("Превышено время ожидания ответа сервера. Попробуйте ещё раз.")
        return WAITING_ASSIGNMENT
    except Exception as e:
        logger.error(f"Error uploading assignment: {e}")
        await update.message.reply_text("Произошла ошибка при загрузке задания. Попробуйте ещё раз.")
        return WAITING_ASSIGNMENT


async def handle_essay_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обрабатывает загрузку файла эссе для анализа.
    Args:
        update (Update): Объект обновления Telegram
        context (ContextTypes.DEFAULT_TYPE): Контекст бота
    Returns:
        int: Следующее состояние разговора
    """
    user_id = update.message.from_user.id

    # Проверка дневного лимита
    if not has_daily_quota(user_id):
        keyboard = get_main_menu_keyboard()
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)
        await update.message.reply_text(
            "Лимит: не более одной рекомендации в день. Попробуйте завтра.",
            reply_markup=reply_markup,
        )
        return MAIN_MENU

    if not update.message.document or (
        not update.message.document.file_name.endswith(".txt")
        and not update.message.document.file_name.endswith(".docx")
    ):
        await update.message.reply_text("Неверный формат файла. Поддерживаются только .txt и .docx.")
        return WAITING_ESSAY

    try:
        file = await update.message.document.get_file()
        file_bytes = await file.download_as_bytearray()
        file_name = update.message.document.file_name

        # Проверка на prompt injection (для текстовых файлов)
        if file_name.endswith(".txt"):
            try:
                text_content = file_bytes.decode("utf-8")
                is_safe, message = prompt_security.check_prompt_injection(text_content)
                if not is_safe:
                    await update.message.reply_text(
                        "Ваш файл содержит подозрительное содержимое и был отклонен системой безопасности."
                    )
                    logger.warning(f"Prompt injection detected in file from user {user_id}: {message}")
                    return WAITING_ESSAY
            except UnicodeDecodeError:
                # Файл не текстовый, пропускаем проверку
                pass

        files = {"file": (file_name, file_bytes)}
        data = {"user_id": str(user_id)}

        await update.message.reply_text("Анализирую ваш текст с учётом задания...")
        response = requests.post(f"{BACKEND_URL}/analyze", files=files, data=data, timeout=120)

        if response.status_code == 200:
            response_data = response.json()
            keyboard = get_main_menu_keyboard()
            reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)

            recommendation = response_data.get("recommendation", "")

            # Преобразуем **bold** в HTML <b>bold</b> для корректного выделения
            def md_bold_to_html(s: str) -> str:
                return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s, flags=re.S)

            recommendation_html = md_bold_to_html(recommendation)
            parts = split_text_for_telegram(recommendation_html, max_len=4096)
            if parts:
                for part in parts[:-1]:
                    await update.message.reply_text(part)
                await update.message.reply_text(parts[-1], reply_markup=reply_markup)
            else:
                await update.message.reply_text("Пустой ответ.", reply_markup=reply_markup)

            # Фиксируем использование лимита
            record_daily_use(user_id)
            return MAIN_MENU
        else:
            # Обработка различных типов ошибок
            try:
                error_data = response.json()
                error_type = error_data.get("error")
                error_message = error_data.get("message", "Произошла ошибка")

                if error_type == "no_assignment":
                    await update.message.reply_text(
                        "Сначала загрузите задание: нажмите «Загрузить задание от преподавателя» и "
                        "отправьте файл (.docx или .txt)."
                    )
                elif error_type == "infected_file":
                    await update.message.reply_text(
                        "Ваш файл содержит потенциально опасное содержимое и был отклонен системой безопасности. "
                        + "Пожалуйста, проверьте файл и попробуйте еще раз."
                    )
                elif error_type == "prompt_injection":
                    await update.message.reply_text(
                        "Обнаружена попытка несанкционированного воздействия на систему. "
                        + "Пожалуйста, проверьте содержимое файла и попробуйте еще раз."
                    )
                elif error_type == "invalid_docx":
                    await update.message.reply_text(
                        "Загруженный DOCX файл поврежден или имеет неверный формат. "
                        + "Пожалуйста, проверьте файл и попробуйте еще раз."
                    )
                elif error_type == "unsupported_format":
                    await update.message.reply_text(
                        "Неподдерживаемый формат файла. Поддерживаются только .txt и .docx."
                    )
                else:
                    await update.message.reply_text(
                        "К сожалению, произошла ошибка при обработке вашего файла. Попробуйте ещё раз."
                    )
            except:
                await update.message.reply_text(
                    "К сожалению, произошла ошибка при обработке вашего файла. Попробуйте ещё раз."
                )
            return WAITING_ESSAY
    except requests.exceptions.Timeout:
        await update.message.reply_text("Превышено время ожидания ответа от сервера. Попробуйте ещё раз.")
        return WAITING_ESSAY
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        await update.message.reply_text(
            "Произошла непредвиденная ошибка. Попробуйте ещё раз или обратитесь к администратору."
        )
        return WAITING_ESSAY


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обрабатывает отмену операции.
    Args:
        update (Update): Объект обновления Telegram
        context (ContextTypes.DEFAULT_TYPE): Контекст бота
    Returns:
        int: Следующее состояние разговора
    """
    keyboard = get_main_menu_keyboard()
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)

    await update.message.reply_text("Операция отменена. Чем еще могу помочь?", reply_markup=reply_markup)
    return MAIN_MENU


async def cancel_without_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обрабатывает отмену без дополнительного сообщения.
    Args:
        update (Update): Объект обновления Telegram
        context (ContextTypes.DEFAULT_TYPE): Контекст бота
    Returns:
        int: Следующее состояние разговора
    """
    keyboard = get_main_menu_keyboard()
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)

    await update.message.reply_text("Чем могу помочь?", reply_markup=reply_markup)
    return MAIN_MENU


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обрабатывает ошибки в работе бота.
    Args:
        update (Update): Объект обновления Telegram
        context (ContextTypes.DEFAULT_TYPE): Контекст бота
    Returns:
        int: Следующее состояние разговора
    """
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

    keyboard = get_main_menu_keyboard()
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)

    await update.message.reply_text(
        "Произошла непредвиденная ошибка. Давайте начнем сначала.",
        reply_markup=reply_markup,
    )
    return MAIN_MENU


def main() -> None:
    """
    Основная функция запуска бота.
    """
    app = Application.builder().token(BOT_TOKEN).defaults(Defaults(parse_mode=ParseMode.HTML)).build()
    app.add_error_handler(error_handler)

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            MAIN_MENU: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_main_menu),
                MessageHandler(filters.Document.ALL, handle_incorrect_action),
            ],
            WAITING_ASSIGNMENT: [
                MessageHandler(filters.Document.ALL, handle_assignment_document),
                MessageHandler(
                    filters.TEXT & ~filters.Regex(f"^{BTN_CANCEL}$"),
                    handle_incorrect_action,
                ),
                MessageHandler(filters.Regex(f"^{BTN_CANCEL}$"), cancel),
            ],
            WAITING_ESSAY: [
                MessageHandler(filters.Document.ALL, handle_essay_document),
                MessageHandler(
                    filters.TEXT & ~filters.Regex(f"^{BTN_CANCEL}$"),
                    handle_incorrect_action,
                ),
                MessageHandler(filters.Regex(f"^{BTN_CANCEL}$"), cancel),
            ],
            WAITING_RATING: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_rating),
                MessageHandler(filters.Regex(f"^{BTN_CANCEL}$"), cancel),
                MessageHandler(filters.Document.ALL, handle_incorrect_action),
            ],
            WAITING_COMMENT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_comment),
                MessageHandler(filters.Document.ALL, handle_incorrect_action),
            ],
        },
        fallbacks=[
            CommandHandler("start", start),
            CommandHandler("cancel", cancel_without_message),
        ],
    )

    app.add_handler(conv_handler)
    logger.info("Бот запущен")
    app.run_polling()


if __name__ == "__main__":
    main()
