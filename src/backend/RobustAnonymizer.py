import logging
import re
from logging.handlers import RotatingFileHandler

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

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


class RobustAnonymizer:
    """Анонимайзер с использованием regex и Presidio"""

    def __init__(self):
        """Инициализирует анонимайзер с regex-паттернами и Presidio (если доступен)."""
        self.patterns = {
            "phone": r"(\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "passport": r"\b(\d{4}\s?\d{6})\b",
            "name": r"\b([А-Я][а-я]+\s[А-Я][а-я]+)\b",
            "credit_card": r"\b(\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4})\b",
            "address": r"\b(ул\.|улица|пр\.|проспект|пер\.|переулок|д\.|дом|кв\.|квартира)\s+[А-Яа-я0-9\s\-]+\b",
        }

        self.presidio_available = False
        try:
            try:
                provider = NlpEngineProvider(
                    conf_file=None,
                    languages=["ru"],
                    nlp_configuration={
                        "nlp_engine_name": "spacy",
                        "models": [{"lang_code": "ru", "model_name": "ru_core_news_md"}],
                    },
                )
                nlp_engine = provider.create_engine()
                self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["ru"])
                logger.info("Presidio initialized with Russian language support")
            except Exception:
                self.analyzer = AnalyzerEngine()
                logger.info("Presidio initialized with default settings")

            self.anonymizer = AnonymizerEngine()
            self.presidio_available = True
            logger.info("Presidio initialized successfully")
        except ImportError as e:
            logger.warning(f"Presidio not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize Presidio: {e}")

    def anonymize_text(self, text):
        """
        Анонимизирует текст с помощью Presidio (если доступен) или regex-шаблонов.
        Args:
            text (str): Исходный текст для анонимизации
        Returns:
            str: Анонимизированный текст
        """
        if not text:
            return text

        if self.presidio_available:
            try:
                results = self.analyzer.analyze(text=text, language="ru")
                if results:
                    anonymized_result = self.anonymizer.anonymize(
                        text=text,
                        analyzer_results=results,
                        operators={
                            "PERSON": {"type": "replace", "new_value": "[ИМЯ]"},
                            "PHONE_NUMBER": {
                                "type": "replace",
                                "new_value": "[ТЕЛЕФОН]",
                            },
                            "EMAIL_ADDRESS": {
                                "type": "replace",
                                "new_value": "[EMAIL]",
                            },
                            "URL": {"type": "replace", "new_value": "[ССЫЛКА]"},
                            "LOCATION": {"type": "replace", "new_value": "[АДРЕС]"},
                            "DATE_TIME": {"type": "replace", "new_value": "[ДАТА]"},
                            "NRP": {"type": "replace", "new_value": "[ГРАЖДАНСТВО]"},
                            "ID": {"type": "replace", "new_value": "[ДОКУМЕНТ]"},
                            "CREDIT_CARD": {"type": "replace", "new_value": "[КАРТА]"},
                            "IP_ADDRESS": {"type": "replace", "new_value": "[IP]"},
                        },
                    )
                    return anonymized_result.text
            except Exception as e:
                logger.warning(f"Presidio failed, using regex anonymizer: {e}")

        anonymized = text
        anonymized = re.sub(self.patterns["phone"], "[ТЕЛЕФОН]", anonymized)
        anonymized = re.sub(self.patterns["email"], "[EMAIL]", anonymized)
        anonymized = re.sub(self.patterns["passport"], "[ПАСПОРТ]", anonymized)
        anonymized = re.sub(self.patterns["name"], "[ИМЯ]", anonymized)
        anonymized = re.sub(self.patterns["credit_card"], "[КАРТА]", anonymized)
        anonymized = re.sub(self.patterns["address"], "[АДРЕС]", anonymized)

        return anonymized

    def extract_student_name(self, text: str, user_context: dict = None) -> str:
        """
        Извлекает имя студента из текста.
        Args:
            text (str): Текст для анализа
            user_context (dict): Контекст пользователя из Telegram
        Returns:
            str: Найденное имя или None
        """
        if user_context and "first_name" in user_context:
            return user_context.get("first_name") + " " + user_context.get("last_name", "")

        # Ищем русские ФИО в начале и конце текста
        name_pattern = r"\b([А-Я][а-я]+\s+[А-Я][а-я]+(?:\s+[А-Я][а-я]+)?)\b"
        start_section = text[:300]
        start_match = re.search(name_pattern, start_section)
        if start_match:
            return start_match.group(1)

        end_section = text[-200:]
        end_match = re.search(name_pattern, end_section)
        if end_match:
            return end_match.group(1)

        return None

    def anonymize_student_name_only(self, text: str, student_name: str) -> str:
        """
        Анонимизирует только имя студента, оставляя другие имена.
        Args:
            text (str): Исходный текст
            student_name (str): Имя студента для анонимизации
        Returns:
            str: Текст с анонимизированным именем студента
        """
        if not text or not student_name:
            return text

        name_parts = student_name.split()
        patterns = []
        if len(name_parts) >= 2:
            # Полное имя (Иван Иванов)
            patterns.append(re.escape(student_name))
            # Имя + фамилия в разных порядках
            patterns.append(re.escape(f"{name_parts[0]} {name_parts[1]}"))
            if len(name_parts) > 2:
                patterns.append(re.escape(f"{name_parts[0]} {name_parts[2]}"))

        anonymized_text = text
        for pattern in patterns:
            anonymized_text = re.sub(pattern, "[СТУДЕНТ]", anonymized_text, flags=re.IGNORECASE)

        return anonymized_text
