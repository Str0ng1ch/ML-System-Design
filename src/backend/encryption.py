import os
from typing import Optional

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class EncryptionManager:
    """Менеджер для шифрования и дешифрования данных с использованием AES-CBC."""

    def __init__(self) -> None:
        """
        Инициализирует менеджер шифрования с ключом из переменных окружения.
        Raises:
            ValueError: Если переменная окружения ENCRYPTION_KEY не установлена
        """
        # Получаем ключ из переменных окружения
        key = os.environ.get("ENCRYPTION_KEY")
        if not key:
            raise ValueError("ENCRYPTION_KEY environment variable is not set")

        # Деривация ключа с помощью PBKDF2
        salt = b"fixed_salt_value_"  # В продакшене используйте уникальную соль для каждого файла
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        self.key = kdf.derive(key.encode())

    def encrypt(self, data: bytes) -> bytes:
        """
        Шифрует данные с использованием AES-CBC.
        Args:
            data (bytes): Данные для шифрования
        Returns:
            bytes: Зашифрованные данные с IV в начале
        """
        # Генерируем случайный IV
        iv = os.urandom(16)
        # Создаем шифр
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        # Дополняем данные до размера кратного 16 байтам
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data) + padder.finalize()
        # Шифруем
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        # Возвращаем IV и зашифрованные данные
        return iv + encrypted_data

    def decrypt(self, encrypted_data: bytes) -> Optional[bytes]:
        """
        Расшифровывает данные, зашифрованные с помощью AES-CBC.
        Args:
            encrypted_data (bytes): Зашифрованные данные с IV в начале
        Returns:
            Optional[bytes]: Расшифрованные данные или None в случае ошибки
        """
        try:
            # Извлекаем IV (первые 16 байт)
            iv = encrypted_data[:16]
            actual_encrypted = encrypted_data[16:]
            # Создаем шифр
            cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            # Расшифровываем
            padded_data = decryptor.update(actual_encrypted) + decryptor.finalize()
            # Убираем дополнение
            unpadder = padding.PKCS7(128).unpadder()
            data = unpadder.update(padded_data) + unpadder.finalize()
            return data
        except Exception:
            return None


# Глобальный экземпляр для использования
encryption_manager = EncryptionManager()
