import json
import os
from pathlib import Path
from typing import List, Optional

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()


class DecryptionManager:
    """Менеджер для расшифровки файлов, зашифрованных EncryptionManager."""

    def __init__(self) -> None:
        """
        Инициализирует менеджер расшифровки с ключом из переменных окружения.
        Raises:
            ValueError: Если переменная окружения ENCRYPTION_KEY не установлена
        """
        key = os.environ.get("ENCRYPTION_KEY")
        if not key:
            raise ValueError("ENCRYPTION_KEY environment variable is not set")

        # Деривация ключа (такая же как в encryption.py)
        salt = b"fixed_salt_value_"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        self.key = kdf.derive(key.encode())

    def decrypt(self, encrypted_data: bytes) -> Optional[bytes]:
        """
        Расшифровывает данные.
        Args:
            encrypted_data (bytes): Зашифрованные данные для расшифровки
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
        except Exception as e:
            print(f"Ошибка при расшифровке: {e}")
            return None

    def decrypt_file(self, file_path: Path) -> Optional[str]:
        """
        Расшифровывает файл и возвращает содержимое.
        Args:
            file_path (Path): Путь к зашифрованному файлу
        Returns:
            Optional[str]: Расшифрованное содержимое файла или None в случае ошибки
        """
        try:
            with open(file_path, "rb") as f:
                encrypted_data = f.read()

            decrypted_data = self.decrypt(encrypted_data)
            if decrypted_data:
                return decrypted_data.decode("utf-8")
            return None
        except Exception as e:
            print(f"Ошибка при чтении файла {file_path}: {e}")
            return None

    def decrypt_and_save(self, encrypted_file_path: Path, output_file_path: Path) -> bool:
        """
        Расшифровывает файл и сохраняет результат.
        Args:
            encrypted_file_path (Path): Путь к зашифрованному файлу
            output_file_path (Path): Путь для сохранения расшифрованного файла
        Returns:
            bool: True если расшифровка успешна, иначе False
        """
        decrypted_content = self.decrypt_file(encrypted_file_path)
        if decrypted_content:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(decrypted_content)
            print(f"Файл расшифрован: {output_file_path}")
            return True
        return False


def find_encrypted_files(base_dir: str) -> List[Path]:
    """
    Находит все зашифрованные файлы в директории.
    Args:
        base_dir (str): Базовая директория для поиска
    Returns:
        List[Path]: Список путей к зашифрованным файлам
    """
    encrypted_files = []
    base_path = Path(base_dir)

    # Ищем файлы с расширениями .enc
    for ext in ["*.enc", "*_verdict.enc", "*_chunks.enc"]:
        for file_path in base_path.rglob(ext):
            encrypted_files.append(file_path)

    return encrypted_files


def main() -> None:
    """Основная функция для расшифровки файлов."""
    # Инициализируем менеджер расшифровки
    try:
        decryption_manager = DecryptionManager()
        print("✅ Менеджер расшифровки инициализирован успешно")
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        return

    # Базовая директория для поиска файлов
    base_directory = "../../data/uploads/1402792053"  # Измените на вашу реальную директорию

    # Находим все зашифрованные файлы
    encrypted_files = find_encrypted_files(base_directory)

    if not encrypted_files:
        print("❌ Зашифрованные файлы не найдены")
        print(f"Искали в директории: {base_directory}")
        print("Проверьте пути и убедитесь, что файлы с расширением .enc существуют")
        return

    print(f"📁 Найдено зашифрованных файлов: {len(encrypted_files)}")

    # Создаем директорию для расшифрованных файлов
    decrypted_dir = Path(base_directory).parent / "decrypted"
    decrypted_dir.mkdir(exist_ok=True)

    # Расшифровываем каждый файл
    for encrypted_file in encrypted_files:
        print(f"\n🔓 Обрабатываем файл: {encrypted_file}")

        # Определяем тип файла по имени
        file_name = encrypted_file.name
        if "_verdict.enc" in file_name:
            output_ext = ".txt"
            file_type = "вердикт"
        elif "_chunks.enc" in file_name:
            output_ext = ".json"
            file_type = "чанки"
        elif file_name.endswith(".enc"):
            output_ext = ".txt"
            file_type = "работа"
        else:
            output_ext = ".txt"
            file_type = "файл"

        # Создаем имя для выходного файла
        output_file_name = encrypted_file.stem  # Убираем .enc
        if output_ext == ".json":
            output_file_name = output_file_name.replace("_chunks", "")
        output_file = decrypted_dir / f"{output_file_name}{output_ext}"

        # Расшифровываем
        success = decryption_manager.decrypt_and_save(encrypted_file, output_file)

        if success:
            print(f"✅ {file_type.capitalize()} успешно расшифрован")

            # Показываем превью содержимого
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    content = f.read()

                if output_ext == ".json":
                    # Для JSON файлов пытаемся красиво отформатировать
                    try:
                        json_data = json.loads(content)
                        preview = (
                            json.dumps(json_data, ensure_ascii=False, indent=2)[:500] + "..."
                            if len(content) > 500
                            else json.dumps(json_data, ensure_ascii=False, indent=2)
                        )
                        print(f"📊 Превью ({len(content)} символов):\n{preview}")
                    except:
                        print(f"📄 Превью ({len(content)} символов):\n{content[:500]}...")
                else:
                    print(f"📄 Превью ({len(content)} символов):\n{content[:500]}...")

            except Exception as e:
                print(f"⚠️ Не удалось прочитать расшифрованный файл: {e}")
        else:
            print(f"❌ Не удалось расшифровать {file_type}")


def test_encryption() -> None:
    """Тестовая функция для проверки работы шифрования/дешифрования."""
    print("\n🧪 Тестирование шифрования...")

    try:
        decryption_manager = DecryptionManager()

        # Тестовый текст
        test_text = "Это тестовый текст для проверки шифрования. Студент: Иван Иванов, телефон: +7 999 123-45-67"

        # Шифруем (имитируем работу encryption_manager)
        import os

        from cryptography.hazmat.primitives import padding
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        iv = os.urandom(16)
        cipher = Cipher(
            algorithms.AES(decryption_manager.key),
            modes.CBC(iv),
            backend=default_backend(),
        )
        encryptor = cipher.encryptor()
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(test_text.encode()) + padder.finalize()
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        final_encrypted = iv + encrypted_data

        # Расшифровываем
        decrypted = decryption_manager.decrypt(final_encrypted)

        if decrypted and decrypted.decode("utf-8") == test_text:
            print("✅ Тест шифрования пройден успешно!")
            print(f"Оригинал: {test_text}")
            print(f"Расшифровано: {decrypted.decode('utf-8')}")
        else:
            print("❌ Тест шифрования не пройден")

    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")


if __name__ == "__main__":
    print("🔓 Скрипт для расшифровки файлов")
    print("=" * 50)

    # Проверяем наличие ключа
    encryption_key = os.environ.get("ENCRYPTION_KEY")
    if not encryption_key:
        print("❌ Переменная окружения ENCRYPTION_KEY не найдена")
        print("Убедитесь, что файл .env существует и содержит ENCRYPTION_KEY")
    else:
        print(f"✅ ENCRYPTION_KEY найден: {encryption_key[:10]}...{encryption_key[-10:]}")

    # Запускаем тест
    test_encryption()

    print("\n" + "=" * 50)
    print("🔍 Поиск и расшифровка файлов...")

    # Запускаем основную функцию
    main()
