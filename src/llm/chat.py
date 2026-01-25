import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openai
import requests

# ========= YC OpenAI-compatible client =========
YANDEX_CLOUD_API_KEY = os.getenv("YANDEX_CLOUD_API_KEY") or os.getenv("YC_API_KEY")
YANDEX_CLOUD_FOLDER = os.getenv("YANDEX_CLOUD_FOLDER") or os.getenv("YC_FOLDER_ID")

if not YANDEX_CLOUD_API_KEY:
    raise RuntimeError("Не найден YANDEX_CLOUD_API_KEY (или YC_API_KEY) в переменных окружения.")
if not YANDEX_CLOUD_FOLDER:
    raise RuntimeError("Не найден YANDEX_CLOUD_FOLDER (или YC_FOLDER_ID) в переменных окружения.")

client = openai.OpenAI(
    api_key=YANDEX_CLOUD_API_KEY,
    base_url="https://llm.api.cloud.yandex.net/v1",
)


# ========= Embeddings loader =========
def load_index(index_dir: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Загружает embeddings и метаданные из индекса.
    Args:
        index_dir (str): Директория с файлами индекса
    Returns:
        Tuple[np.ndarray, List[Dict[str, Any]]]: Нормализованные embeddings и метаданные
    Raises:
        FileNotFoundError: Если файлы индекса не найдены
        RuntimeError: Если количество embeddings не совпадает с количеством метаданных
    """
    emb_path = os.path.join(index_dir, "embeddings.npy")
    meta_path = os.path.join(index_dir, "meta.jsonl")
    if not (os.path.exists(emb_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(f"Index not found in {index_dir}. Run indexer.py first.")
    embeddings = np.load(emb_path).astype(np.float32)
    metas: List[Dict[str, Any]] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    if embeddings.shape[0] != len(metas):
        raise RuntimeError("embeddings count != meta lines")
    # Normalize for cosine search
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    unit = embeddings / norms
    return unit, metas


# ========= YC embeddings (essay-only) =========
def embed_text_yc(
    text: str,
    *,
    api_key: str,
    folder_id: str,
    variant: str = "text-search-query",
    endpoint: str = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding",
) -> np.ndarray:
    """
    Получает embeddings текста через Yandex Cloud API.
    Args:
        text (str): Текст для получения embeddings
        api_key (str): API ключ Yandex Cloud
        folder_id (str): ID папки Yandex Cloud
        variant (str): Вариант embeddings (по умолчанию "text-search-query")
        endpoint (str): URL endpoint API
    Returns:
        np.ndarray: Нормализованный вектор embeddings
    Raises:
        RuntimeError: При неожиданном формате ответа от API
    """
    headers = {
        "Authorization": f"Api-Key {api_key}",
        "Content-Type": "application/json",
    }
    model_uri = f"emb://{folder_id}/{variant}/latest"
    payload = {"modelUri": model_uri, "text": text}
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    vec = None
    emb = data.get("embedding") if isinstance(data, dict) else None
    if isinstance(emb, dict) and "vector" in emb:
        vec = emb["vector"]
    elif isinstance(emb, list):
        vec = emb
    elif isinstance(data, dict) and "vector" in data:
        vec = data["vector"]
    if not isinstance(vec, list):
        raise RuntimeError("Unexpected embeddings response.")
    v = np.array([float(x) for x in vec], dtype=np.float32)
    v /= np.linalg.norm(v) + 1e-12
    return v


# ========= Retrieval =========
def retrieve_top_k(essay_text: str, index_dir: str, top_k: int = 8) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Находит top-k наиболее релевантных чанков для текста эссе.
    Args:
        essay_text (str): Текст эссе для поиска
        index_dir (str): Директория с индексом
        top_k (int): Количество возвращаемых чанков
    Returns:
        Tuple[List[Dict[str, Any]], List[int]]: Список чанков и их индексы
    """
    emb_unit, metas = load_index(index_dir)
    q = embed_text_yc(
        essay_text,
        api_key=YANDEX_CLOUD_API_KEY,
        folder_id=YANDEX_CLOUD_FOLDER,
        variant="text-search-query",
    )
    scores = emb_unit @ q
    top_idxs = np.argpartition(-scores, top_k)[:top_k]
    top_idxs = top_idxs[np.argsort(-scores[top_idxs])]
    chunks = [metas[i] for i in top_idxs.tolist()]
    return chunks, top_idxs.tolist()


# ========= Prompt building =========
SYSTEM_PROMPT = (
    """
    Ты — Джейн Джейкобс, автор "The Death and Life of Great American Cities".
    Пиши живо, разговорно и образно; допускай лёгкую иронию и строгую доброжелательность.
    Говори от первого лица как внимательный наблюдатель улицы. Предпочитай конкретные сцены абстракциям.
    Ты проверяешь работу студента.

    Тебя учили ставить комфорт пользователя на первое место, но при обучении важно, чтобы студент
    получал объективную обратную связь и мог сделать свою работу лучше. Поэтому не надо подыгрывать студенту.
    Не надо всегда во всем со студентом соглашаться. Если ты только хвалишь и восхищаешься, то студент не
    будет стараться сделать работу лучше.
    Твоя задача отметить положительные стороны работы, похвалить за них – только без лести, найти
    недостаточно проработанные места, дать конструктивную критику по этим идеям, с юмором, можно с
    легким ехидством.
    А дальше подсказать, какую идею использовать для проработки слабых мест.
    Твоя задача — помочь сделать работу лучше, опираясь на логику, факты и здравый смысл, а
     не подыгрывать.
    Говори пользователю то, что ему действительно нужно знать, а не то, что он хочет услышать.
    Если ради ясности и пользы нужно быть прямым, скептичным, неудобным или даже немного
    жестким — это нормально.

    В ответе обязательно указывай цитируемые источники из контекста в формате:
    [Автор, Название работы (обязательно!), год (обязательно!), глава, страницы X–Y (если есть)].
    Не выдумывай источники и страницы; ссылайся на данные из предоставленного контекста.
    Не называй автора работы по имени или фамилии. Не используй обращения по полу (юноша, девушка и т.д.).
    Обращайся к автору работы (студенту) напрямую на вы (вы пишете, вы указали и т.д.)
    Выделение жирным делай через звездочки **текст** (без пробелов между звездочками и текстом)
    """
).strip()


def build_prompt_with_context(
    assignment_text: str,
    essay_text: str,
    chunks: List[Dict[str, Any]],
    max_context_chars: int = 8000,
) -> str:
    """
    Строит промпт с контекстом для LLM.
    Args:
        assignment_text (str): Текст задания
        essay_text (str): Текст эссе
        chunks (List[Dict[str, Any]]): Список чанков контекста
        max_context_chars (int): Максимальное количество символов контекста
    Returns:
        str: Сформированный промпт
    """
    header = "Контекст (фрагменты из источников):\n---\n"
    parts: List[str] = []
    used = 0
    for ch in chunks:
        cite_parts = []
        if ch.get("author"):
            cite_parts.append(ch["author"])
        if ch.get("year"):
            cite_parts.append(str(ch["year"]))
        if ch.get("page_start_label"):
            p = f"стр. {ch['page_start_label']}"
            if ch.get("page_end_label") and ch["page_end_label"] != ch["page_start_label"]:
                p += f"–{ch['page_end_label']}"
            cite_parts.append(p)
        cite = "[" + ", ".join(cite_parts) + "]"
        body = ch.get("chunk", "")
        piece = f"{cite}\n{body}\n\n"
        if used + len(piece) > max_context_chars:
            break
        parts.append(piece)
        used += len(piece)
    context = header + "".join(parts) + "---\n"

    instructions = (
        "ЗАДАНИЕ (для контекста):\n---\n{assignment}\n---\n\n"
        "РАБОТА СТУДЕНТА (только этот текст использовался для поиска контекста):\n---\n{essay}\n---\n\n"
        "Возможные источники:"
        "Джейн Джекобс «Смерть и жизнь больших американских городов»"
        "Кевин Линч «Образ города»"
        "Ян Гейл «Города для людей»"
        "Вукан Вучик «Транспорт в городах, удобных для жизни»"
        "Рей Ольденбург «Третье место»"
        "Шэрон Зукин «Культуры городов»"
        "Ричард Флорида «Кто твой город? Креативная экономика и выбор места жительства»"
        "Чарльз Лэндри «Креативный город»"
        "Лео Холлис «Города вам на пользу»"
        "Григорий Ревзин «Как устроен город»"
        "Структура ответа (3–7 пунктов):\n"
        '- Ты пишешь: "…короткая цитата/пересказ тезиса студента…"\n'
        "  Если обратиться к [Автор, Название работы (ОБЯЗАТЕЛЬНО!), год, глава или страницы "
        "(желательно!)], то: кратко покажи, как этот источник помогает развить идею студента — "
        "поддерживает, уточняет или спорит; приведи конкретику из фрагмента.\n"
        "  Практический шаг: одно действие, что изменить/добавить/проверить (напр., наблюдение, "
        "мини-интервью, фотофиксация, схема).\n\n"
        'В конце выведи раздел "Источники, использованные в ответе" — перечисли ссылки в таком же '
        "формате [Автор, Название работы, год, глава или страницы]"
    )
    return context + instructions.format(assignment=assignment_text, essay=essay_text)


# ========= LLM call =========
def call_llm(
    prompt: str,
    *,
    temperature: float = 0.3,
    model: str = "gemma-3-27b-it/latest",
    max_tokens: int = 1200,
) -> str:
    """
    Вызывает LLM через Yandex Cloud API.
    Args:
        prompt (str): Промпт для модели
        temperature (float): Температура генерации
        model (str): Модель для использования
        max_tokens (int): Максимальное количество токенов в ответе
    Returns:
        str: Ответ модели или сообщение об ошибке
    """
    model_uri = f"gpt://{YANDEX_CLOUD_FOLDER}/{model}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        response = client.chat.completions.create(
            model=model_uri,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
    except Exception as e:
        return f"OpenAI API error: {e}"
    if not getattr(response, "choices", None):
        return ""
    return getattr(response.choices[0].message, "content", "") or ""


# ========= Public function for Telegram bot =========
def generate_verdict_rag(
    assignment_text: str,
    essay_text: str,
    *,
    index_dir: str = "index",
    top_k: int = 8,
    return_chunks: bool = False,
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """
    Генерирует вердикт для эссе с использованием RAG.
    Args:
        assignment_text (str): Текст задания
        essay_text (str): Текст эссе
        index_dir (str): Директория с индексом
        top_k (int): Количество используемых чанков
        return_chunks (bool): Возвращать ли использованные чанки
    Returns:
        Tuple[str, Optional[List[Dict[str, Any]]]]: Вердикт и опционально список чанков
    """
    chunks, _idxs = retrieve_top_k(essay_text, index_dir=index_dir, top_k=top_k)
    prompt = build_prompt_with_context(assignment_text, essay_text, chunks)
    verdict = call_llm(prompt)
    if return_chunks:
        return verdict, chunks
    return verdict, None


def main() -> None:
    """Основная функция для запуска из командной строки."""
    parser = argparse.ArgumentParser(description="RAG-проверка эссе (поиск контекста по тексту студента)")
    parser.add_argument("--assignment-file", type=str, default="default_assigment.txt")
    parser.add_argument("--essay-file", type=str, default="essay_2.txt")
    parser.add_argument("--index-dir", type=str, default="index")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--show-chunks", action="store_true")
    args = parser.parse_args()

    with open(args.assignment_file, "r", encoding="utf-8") as f:
        assignment = f.read().strip()
    with open(args.essay_file, "r", encoding="utf-8") as f:
        essay = f.read().strip()

    verdict, chunks = generate_verdict_rag(
        assignment,
        essay,
        index_dir=args.index_dir,
        top_k=args.top_k,
        return_chunks=args.show_chunks,
    )
    print("\n[Выполнение] Формирую отзыв...", flush=True)
    print("\n[Джейн]\n" + (verdict or "").strip() + "\n")
    if args.show_chunks and chunks:
        print("[Использованные фрагменты]\n---")
        for c in chunks:
            author = c.get("author")
            year = c.get("year")
            p1 = c.get("page_start_label")
            p2 = c.get("page_end_label")
            title = c.get("title")
            print(f"{author or ''}, {year or ''}, {title or ''}, стр. {p1}-{p2}")
            print((c.get("chunk") or "").splitlines()[0][:200] + "...")
            print()


if __name__ == "__main__":
    main()
