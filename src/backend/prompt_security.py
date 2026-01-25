import logging
import os
from typing import Tuple

import requests

logger = logging.getLogger(__name__)


class PromptSecurity:
    """Класс для проверки безопасности промптов с использованием Lakera Guard API."""

    def __init__(self) -> None:
        """Инициализирует класс с API ключом Lakera Guard."""
        self.api_key = os.environ.get("LAKERA_GUARD_API_KEY")
        self.api_url = "https://api.lakera.ai/v2/guard"

    def check_prompt_injection(self, text: str) -> Tuple[bool, str]:
        """
        Проверяет текст на наличие инъекций промптов.
        Args:
            text (str): Текст для проверки
        Returns:
            Tuple[bool, str]: (Результат проверки, Сообщение)
        """
        if not self.api_key:
            logger.warning("Lakera Guard API key not set")
            return True, "API key not configured"

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {"messages": [{"content": text, "role": "user"}]}

            response = requests.post(self.api_url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            result = response.json()

            if result.get("flagged"):
                return False, "Prompt injection detected"
            return True, "No prompt injection detected"
        except Exception as e:
            logger.error(f"Error checking prompt injection: {e}")
            return False, f"Error checking prompt security: {e}"


# Глобальный экземпляр
prompt_security = PromptSecurity()
