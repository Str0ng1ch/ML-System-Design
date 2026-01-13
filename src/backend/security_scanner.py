import logging
import os
import tempfile
from typing import Tuple

from pyclamd import ClamdAgnostic

logger = logging.getLogger(__name__)


class SecurityScanner:
    """Класс для сканирования файлов на наличие вредоносного ПО с использованием ClamAV."""

    def __init__(self) -> None:
        """Инициализирует сканер ClamAV."""
        try:
            self.cd = ClamdAgnostic()
            self.available = True
        except Exception as e:
            logger.warning(f"ClamAV not available: {e}")
            self.available = False

    def scan_file(self, file_content: bytes, filename: str) -> Tuple[bool, str]:
        """
        Сканирует файл на наличие вредоносного ПО.
        Args:
            file_content (bytes): Содержимое файла для сканирования
            filename (str): Имя файла (для определения расширения)
        Returns:
            Tuple[bool, str]: (Результат сканирования, Сообщение)
        """
        if not self.available:
            return True, "ClamAV not available"

        try:
            # Создаем временный файл для сканирования
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name

            # Сканируем файл
            scan_result = self.cd.scan_file(temp_path)

            # Удаляем временный файл
            os.unlink(temp_path)

            if scan_result is not None:
                return False, f"File is infected: {scan_result}"
            return True, "File is clean"
        except Exception as e:
            return False, f"Error scanning file: {e}"


# Глобальный экземпляр сканера
security_scanner = SecurityScanner()
