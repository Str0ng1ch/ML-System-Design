FROM python:3.12-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Копируем исходный код
COPY src/ .

# Создаем директории для данных
RUN mkdir -p /app/data/uploads /app/data/feedback

ENV PYTHONPATH=/app

# Открываем порт для backend
EXPOSE 5000

# Запускаем оба процесса
CMD python backend/backend.py & python service-bot/bot1.py
