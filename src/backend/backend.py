import io
import json
import logging
import os
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from zipfile import BadZipFile, is_zipfile

from docx import Document
from encryption import encryption_manager
from flask import Flask, jsonify, request
from prompt_security import prompt_security
from RobustAnonymizer import RobustAnonymizer
from security_scanner import security_scanner

from llm.chat import generate_verdict_rag

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("/app/data/app.log", maxBytes=100000, backupCount=3),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
app = Flask(__name__)
anonymizer = RobustAnonymizer()


def read_docx(file) -> str:
    """
    Читает содержимое DOCX-файла.
    Args:
        file: File-объект DOCX-документа
    Returns:
        str: Текст документа
    """
    doc = Document(io.BytesIO(file.read()))
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])


# def save_uploaded_file(user_id: int, file: io.BytesIO, filename) -> str:
#     """
#         Сохраняет загруженный файл в директорию пользователя.
#         Args:
#             user_id: ID пользователя
#             file: File-объект
#             filename: Имя файла
#         Returns:
#             str: Путь к сохраненному файлу
#         """
#     user_dir = os.path.join('/app/data/uploads', str(user_id))
#     os.makedirs(user_dir, exist_ok=True)
#
#     # Генерируем уникальное имя файла с timestamp
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     file_extension = os.path.splitext(filename)[1]
#     new_filename = f"{timestamp}_{uuid.uuid4().hex}{file_extension}"
#     file_path = os.path.join(user_dir, new_filename)
#
#     file.save(file_path)
#     return file_path


def get_user_dir(user_id: str) -> str:
    """
    Возвращает путь к директории пользователя.
    Args:
        user_id: ID пользователя
    Returns:
        str: Путь к директории
    """
    d = os.path.join("/app/data/uploads", str(user_id))
    os.makedirs(d, exist_ok=True)
    return d


def get_assignment_path(user_id: str) -> str:
    """
    Возвращает путь к файлу с заданием пользователя.
    Args:
        user_id: ID пользователя
    Returns:
        str: Путь к файлу задания
    """
    return os.path.join(get_user_dir(user_id), "assignment.txt")


def save_text_essay(user_id: str, text: str) -> str:
    """
    Сохраняет текст эссе в зашифрованном виде.
    Args:
        user_id: ID пользователя
        text: Текст эссе
    Returns:
        str: Путь к сохраненному файлу
    """
    user_dir = get_user_dir(user_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"{timestamp}_{uuid.uuid4().hex}.enc"
    file_path = os.path.join(user_dir, new_filename)

    # Шифруем текст перед сохранением
    encrypted_data = encryption_manager.encrypt(text.encode("utf-8"))
    with open(file_path, "wb") as f:
        f.write(encrypted_data)
    return file_path


# Модифицируем функцию upload_assignment
@app.route("/assignment", methods=["POST"])
def upload_assignment():
    """Обрабатывает загрузку файла с заданием."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    user_id = request.form.get("user_id", "unknown")

    if not file.filename.endswith((".txt", ".docx")):
        return jsonify({"error": "Unsupported file type"}), 400

    if file.filename.endswith(".docx") and not is_zipfile(file):
        return jsonify({"error": "Uploaded DOCX file is invalid or corrupted"}), 400

    try:
        file.seek(0)
        file_content = file.read()

        # Антивирусная проверка
        is_clean, message = security_scanner.scan_file(file_content, file.filename)
        if not is_clean:
            logger.warning(f"Infected file rejected: {message}")
            return jsonify({"error": "File rejected by security scanner"}), 400

        # Чтение и обработка содержимого
        file.seek(0)
        if file.filename.endswith(".txt"):
            text = file.read().decode("utf-8")
        else:
            text = read_docx(file)

        # Шифрование и сохранение
        assignment_path = get_assignment_path(user_id)
        encrypted_data = encryption_manager.encrypt(text.encode("utf-8"))
        with open(assignment_path, "wb") as f:
            f.write(encrypted_data)

        return jsonify({"status": "success", "message": "Assignment saved"}), 200
    except BadZipFile:
        return jsonify({"error": "The uploaded file is not a valid DOCX file"}), 400
    except Exception as e:
        logger.error(f"Error saving assignment: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/assignment/status", methods=["GET"])
def assignment_status():
    """Проверяет наличие загруженного задания для пользователя."""
    user_id = request.args.get("user_id", "unknown")
    has_assignment = os.path.exists(get_assignment_path(user_id))
    return jsonify({"has_assignment": has_assignment})


@app.route("/analyze", methods=["POST"])
def analyze_document():
    """Анализирует загруженный документ и возвращает рекомендации."""
    if "file" not in request.files:
        return jsonify({"error": "no_file", "message": "Файл не был предоставлен"}), 400

    file = request.files["file"]
    user_id = request.form.get("user_id", "unknown")

    # Требуем наличие сохранённого задания
    if not os.path.exists(get_assignment_path(user_id)):
        return jsonify(
            {
                "error": "no_assignment",
                "message": "Для этого пользователя не найдено задание",
            }
        ), 400

    # Проверка расширения файла
    if not file.filename.endswith((".txt", ".docx")):
        return jsonify({"error": "unsupported_format", "message": "Неподдерживаемый формат файла"}), 400

    # Проверка, что файл является ZIP-архивом (для DOCX)
    if file.filename.endswith(".docx") and not is_zipfile(file):
        return jsonify(
            {
                "error": "invalid_docx",
                "message": "Загруженный DOCX файл поврежден или имеет неверный формат",
            }
        ), 400

    try:
        file.seek(0)
        if file.filename.endswith(".txt"):
            essay_text = file.read().decode("utf-8")
        elif file.filename.endswith(".docx"):
            essay_text = read_docx(file)
        else:
            return jsonify(
                {
                    "error": "unsupported_format",
                    "message": "Неподдерживаемый формат файла",
                }
            ), 400

        # Антивирусная проверка
        is_clean, scan_message = security_scanner.scan_file(file.getvalue(), file.filename)
        if not is_clean:
            logger.warning(f"Infected file rejected from user {user_id}: {scan_message}")
            return jsonify(
                {
                    "error": "infected_file",
                    "message": "Файл содержит потенциально опасное содержимое и был отклонен",
                }
            ), 400

        # Проверка на prompt injection
        is_safe, injection_message = prompt_security.check_prompt_injection(essay_text)
        if not is_safe:
            logger.warning(f"Prompt injection detected from user {user_id}: {injection_message}")
            return jsonify(
                {
                    "error": "prompt_injection",
                    "message": "Обнаружена попытка несанкционированного воздействия на систему",
                }
            ), 400

        # Чтение задания
        with open(get_assignment_path(user_id), "rb") as f:
            encrypted_data = f.read()
        assignment_text = encryption_manager.decrypt(encrypted_data).decode("utf-8")

        student_name = anonymizer.extract_student_name(essay_text)
        anon_essay_text = anonymizer.anonymize_student_name_only(essay_text, student_name)
        anon_assignment_text = anonymizer.anonymize_text(assignment_text)

        # Генерация рекомендаций
        verdict, chunks = generate_verdict_rag(
            anon_assignment_text,
            anon_essay_text,
            index_dir="/app/llm/index",
            top_k=8,
            return_chunks=True,
        )

        # Сохраняем оригинальные данные (зашифрованными)
        saved_txt_path = save_text_essay(user_id, anon_essay_text)
        base, _ext = os.path.splitext(saved_txt_path)

        # Шифруем и сохраняем вердикт
        verdict_path = f"{base}_verdict.enc"
        encrypted_verdict = encryption_manager.encrypt(verdict.encode("utf-8"))
        with open(verdict_path, "wb") as vf:
            vf.write(encrypted_verdict)

        # Шифруем и сохраняем чанки
        chunks_path = f"{base}_chunks.enc"
        encrypted_chunks = encryption_manager.encrypt(json.dumps(chunks).encode("utf-8"))
        with open(chunks_path, "wb") as cf:
            cf.write(encrypted_chunks)

        return jsonify({"recommendation": verdict})

    except BadZipFile:
        return jsonify(
            {
                "error": "invalid_docx",
                "message": "Загруженный DOCX файл поврежден или имеет неверный формат",
            }
        ), 400
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return jsonify(
            {
                "error": "processing_error",
                "message": "Произошла внутренняя ошибка при обработке файла",
            }
        ), 500


@app.route("/feedback", methods=["POST"])
def save_feedback():
    """Сохраняет пользовательский фидбэк."""
    data = request.json
    user_id = data.get("user_id", "unknown")
    rating = data.get("rating", "")
    comment = data.get("comment", "")

    feedback_dir = "/app/data/feedback"
    os.makedirs(feedback_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} | User: {user_id} | Rating: {rating} | Comment: {comment}\n"

    feedback_file = os.path.join(feedback_dir, f"feedback_{datetime.now().strftime('%Y%m')}.txt")

    try:
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except IOError as e:
        logger.error(f"Failed to write feedback: {e}")
        return jsonify({"status": "error", "message": "Failed to save feedback"}), 500

    return jsonify({"status": "success"})


@app.route("/health", methods=["GET"])
def health_check():
    """Проверка статуса работы сервиса."""
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    os.makedirs("/app/data/uploads", exist_ok=True)
    os.makedirs("/app/data/feedback", exist_ok=True)
    logger.info("Ensuring directories exist: /app/data/uploads, /app/data/feedback")

    app.run(host="0.0.0.0", port=5000)
