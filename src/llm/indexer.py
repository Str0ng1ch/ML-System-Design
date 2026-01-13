import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from pypdf import PdfReader
from tqdm import tqdm


@dataclass
class PageInfo:
    """Информация о странице документа."""

    file_page_index: int  # zero-based index within the file
    printed_label: Optional[str]  # human-visible label, if available; else None
    text: str
    chapter_titles: List[str]


@dataclass
class ChapterInfo:
    """Информация о главе документа."""

    title: str
    level: int  # 0 for top-level, increasing with depth
    start_page_index: int  # zero-based index of the page where this chapter starts


@dataclass
class DocumentSource:
    """Источник документа с метаданными и содержимым."""

    file_path: str
    author: Optional[str]
    year: Optional[int]
    title: Optional[str]
    pages: List[PageInfo]
    chapters: List[ChapterInfo]


@dataclass
class Chunk:
    """Чанк текста из документа."""

    text: str
    page_start_index: int
    page_end_index: int
    page_start_label: str
    page_end_label: str
    chapter_titles: List[str]
    file_path: str
    author: Optional[str]
    year: Optional[int]
    title: Optional[str]


def _extract_pdf_metadata(reader) -> Tuple[Optional[str], Optional[str]]:
    """
    Извлекает метаданные из PDF документа.
    Args:
        reader: PdfReader объект
    Returns:
        Tuple[Optional[str], Optional[str]]: (автор, название)
    """
    try:
        info = reader.metadata
    except Exception:
        info = None
    author = None
    title = None
    if info:
        try:
            author = getattr(info, "author", None)
            if not author and hasattr(info, "get"):
                author = info.get("/Author")
        except Exception:
            author = None
        try:
            title = getattr(info, "title", None)
            if not title and hasattr(info, "get"):
                title = info.get("/Title")
        except Exception:
            title = None
    return author, title


def _parse_year_from_filename(file_path: str) -> Optional[int]:
    """
    Пытается извлечь год из имени файла.
    Args:
        file_path (str): Путь к файлу
    Returns:
        Optional[int]: Год или None
    """
    name = os.path.basename(file_path)
    # Look for a 4-digit year between 1800-2099
    matches = re.findall(r"(?<!\d)(18\d{2}|19\d{2}|20\d{2})(?!\d)", name)
    for m in matches:
        try:
            year = int(m)
            if 1800 <= year <= 2099:
                return year
        except Exception:
            pass
    return None


def _get_page_label(reader, index: int) -> Optional[str]:
    """
    Получает метку страницы PDF.
    Args:
        reader: PdfReader объект
        index (int): Индекс страницы
    Returns:
        Optional[str]: Метка страницы или None
    """
    try:
        # pypdf exposes get_page_label
        label = reader.get_page_label(index)
        return str(label) if label is not None else None
    except Exception:
        return None


def _flatten_outline(reader) -> List[Tuple[str, int, int]]:
    """
    Извлекает и упрощает структуру оглавления PDF.
    Args:
        reader: PdfReader объект
    Returns:
        List[Tuple[str, int, int]]: Список (название, индекс страницы, уровень)
    """
    outline = []
    try:
        raw_outline = getattr(reader, "outline", None)
        if raw_outline is None:
            raw_outline = getattr(reader, "outlines", None)

        def walk(items, level=0):
            for item in items:
                try:
                    if isinstance(item, list):
                        walk(item, level)
                        continue
                    title = getattr(item, "title", None) or str(item.title)
                    pnum = None
                    try:
                        # pypdf destination-based page number
                        pnum = reader.get_destination_page_number(item)
                    except Exception:
                        page = getattr(item, "page", None)
                        if page is not None:
                            try:
                                pnum = reader.get_page_number(page)
                            except Exception:
                                pnum = None
                    if pnum is not None:
                        outline.append((str(title).strip(), int(pnum), int(level)))
                except Exception:
                    continue
    except Exception:
        pass
    # Deduplicate by (title, page, level) while preserving order
    seen = set()
    result = []
    for t, p, l in outline:
        key = (t, p, l)
        if key not in seen:
            seen.add(key)
            result.append((t, p, l))
    # Sort by page then by level to ensure consistent ranges
    result.sort(key=lambda x: (x[1], x[2]))
    return result


def _chapters_to_page_coverage(num_pages: int, flat_outline: List[Tuple[str, int, int]]) -> Dict[int, List[str]]:
    """
    Создает покрытие глав по страницам.
    Args:
        num_pages (int): Количество страниц
        flat_outline (List[Tuple[str, int, int]]): Упрощенное оглавление
    Returns:
        Dict[int, List[str]]: Словарь {индекс_страницы: список_названий_глав}
    """
    if not flat_outline:
        return {i: [] for i in range(num_pages)}
    starts = [(p, t, lvl) for (t, p, lvl) in [(t, p, l) for (t, p, l) in flat_outline]]
    # Build a list of (start_page, end_page_exclusive, title)
    ranges: List[Tuple[int, int, str]] = []
    for idx, (start_p, title, _lvl) in enumerate(starts):
        end_p = starts[idx + 1][0] if idx + 1 < len(starts) else num_pages
        start_p = max(0, min(start_p, num_pages))
        end_p = max(start_p, min(end_p, num_pages))
        if start_p < end_p:
            ranges.append((start_p, end_p, title))
    coverage: Dict[int, List[str]] = {i: [] for i in range(num_pages)}
    for start_p, end_p, title in ranges:
        for p in range(start_p, end_p):
            coverage[p].append(title)
    return coverage


def parse_pdf(file_path: str) -> DocumentSource:
    """
    Парсит PDF файл и извлекает структурированную информацию.
    Args:
        file_path (str): Путь к PDF файлу
    Returns:
        DocumentSource: Структурированная информация о документе
    Raises:
        RuntimeError: Если PdfReader недоступен
    """
    if PdfReader is None:
        raise RuntimeError("PdfReader is not available. Please install pypdf or PyPDF2.")
    reader = PdfReader(file_path)
    num_pages = len(reader.pages)

    pdf_author, pdf_title = _extract_pdf_metadata(reader)
    inferred_year = _parse_year_from_filename(file_path)

    flat_outline = _flatten_outline(reader)
    coverage = _chapters_to_page_coverage(num_pages, flat_outline)

    pages: List[PageInfo] = []
    for i in range(num_pages):
        try:
            page = reader.pages[i]
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
        except Exception:
            text = ""
        printed_label = _get_page_label(reader, i)
        chapter_titles = coverage.get(i, [])
        pages.append(
            PageInfo(
                file_page_index=i,
                printed_label=printed_label,
                text=text,
                chapter_titles=chapter_titles,
            )
        )

    # Heuristic: author/title from metadata; if missing, try filename
    author = pdf_author
    title = pdf_title
    if not title:
        base = os.path.splitext(os.path.basename(file_path))[0]
        title = base

    return DocumentSource(
        file_path=file_path,
        author=author,
        year=inferred_year,
        title=title,
        pages=pages,
        chapters=[ChapterInfo(title=t, level=l, start_page_index=p) for (t, p, l) in flat_outline],
    )


def _ensure_djvu_tools_available() -> None:
    """
    Проверяет доступность инструментов djvulibre.
    Raises:
        RuntimeError: Если инструменты не установлены
    """
    for tool in ["djvused", "ddjvu"]:
        try:
            subprocess.run(
                [tool, "--help"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "DJVU support requires 'djvulibre-bin' (tools djvused, ddjvu) to be installed and available in PATH."
            )


def _djvu_num_pages(file_path: str) -> int:
    """
    Получает количество страниц в DJVU файле.
    Args:
        file_path (str): Путь к DJVU файлу
    Returns:
        int: Количество страниц
    Raises:
        RuntimeError: При ошибке получения количества страниц
    """
    _ensure_djvu_tools_available()
    try:
        # djvused -e 'n' file.djvu  => prints a number of pages
        out = subprocess.check_output(["djvused", "-e", "n", file_path], stderr=subprocess.STDOUT)
        txt = out.decode("utf-8", errors="ignore").strip()
        return int(txt)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get number of pages from DJVU: {e.output.decode('utf-8', errors='ignore')}")
    except Exception as e:
        raise RuntimeError(f"Failed to get number of pages from DJVU: {e}")


def _djvu_extract_page_text(file_path: str, page_one_based: int) -> str:
    """
    Извлекает текст из страницы DJVU файла.
    Args:
        file_path (str): Путь к DJVU файлу
        page_one_based (int): Номер страницы (начиная с 1)
    Returns:
        str: Текст страницы
    Raises:
        RuntimeError: При ошибке извлечения текста
    """
    _ensure_djvu_tools_available()
    try:
        # ddjvu can output text for specific page to stdout when target is '-'
        out = subprocess.check_output(
            ["ddjvu", "-format=txt", f"-page={page_one_based}", file_path, "-"],
            stderr=subprocess.STDOUT,
        )
        return out.decode("utf-8", errors="ignore")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to extract text from DJVU page {page_one_based}: {e.output.decode('utf-8', errors='ignore')}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from DJVU page {page_one_based}: {e}")


def parse_djvu(file_path: str) -> DocumentSource:
    """
    Парсит DJVU файл и извлекает структурированную информацию.
    Args:
        file_path (str): Путь к DJVU файлу
    Returns:
        DocumentSource: Структурированная информация о документе
    """
    num_pages = _djvu_num_pages(file_path)
    pages: List[PageInfo] = []
    for i in range(1, num_pages + 1):
        try:
            text = _djvu_extract_page_text(file_path, i)
        except Exception:
            text = ""
        printed_label = str(i)
        chapter_titles: List[str] = []
        pages.append(
            PageInfo(
                file_page_index=i - 1,
                printed_label=printed_label,
                text=(text or ""),
                chapter_titles=chapter_titles,
            )
        )

    inferred_year = _parse_year_from_filename(file_path)
    base = os.path.splitext(os.path.basename(file_path))[0]
    title = base
    author: Optional[str] = None

    return DocumentSource(
        file_path=file_path,
        author=author,
        year=inferred_year,
        title=title,
        pages=pages,
        chapters=[],
    )


def _normalize_label(label: Optional[str], fallback_index_one_based: int) -> str:
    """
    Нормализует метку страницы.
    Args:
        label (Optional[str]): Метка страницы
        fallback_index_one_based (int): Резервный номер страницы
    Returns:
        str: Нормализованная метка
    """
    if label is None:
        return str(fallback_index_one_based)
    try:
        return str(label)
    except Exception:
        return str(fallback_index_one_based)


def chunk_document(
    doc: DocumentSource,
    chunk_chars: int = 2000,
    overlap_chars: int = 200,
) -> List[Chunk]:
    """
    Разбивает документ на чанки.
    Args:
        doc (DocumentSource): Документ для разбиения
        chunk_chars (int): Размер чанка в символах
        overlap_chars (int): Размер перекрытия между чанками
    Returns:
        List[Chunk]: Список чанков
    """
    chunks: List[Chunk] = []
    buffer: List[Tuple[int, str]] = []  # list of (page_index, text_segment)
    buffer_char_len = 0
    buffer_start_page = 0

    def flush(end_page_inclusive: int):
        nonlocal buffer, buffer_char_len, buffer_start_page
        if not buffer:
            return
        text = "".join(seg for _, seg in buffer).strip()
        if not text:
            buffer = []
            buffer_char_len = 0
            return
        page_indices = [pi for pi, _ in buffer]
        start_idx = min(page_indices)
        end_idx = max(page_indices)
        start_label = _normalize_label(doc.pages[start_idx].printed_label, start_idx + 1)
        end_label = _normalize_label(doc.pages[end_idx].printed_label, end_idx + 1)
        # Chapters covered by any page in the chunk
        chapters_set = []
        seen = set()
        for pi in range(start_idx, end_idx + 1):
            for ch in doc.pages[pi].chapter_titles:
                if ch not in seen:
                    seen.add(ch)
                    chapters_set.append(ch)
        header_lines: List[str] = []
        header_lines.append(f"Pages: {start_label}–{end_label} (file {start_idx + 1}–{end_idx + 1})")
        if chapters_set:
            header_lines.append("Chapters: " + "; ".join(chapters_set))
        header = "\n".join(header_lines) + "\n\n"
        chunk_text = header + text
        chunks.append(
            Chunk(
                text=chunk_text,
                page_start_index=start_idx,
                page_end_index=end_idx,
                page_start_label=start_label,
                page_end_label=end_label,
                chapter_titles=chapters_set,
                file_path=doc.file_path,
                author=doc.author,
                year=doc.year,
                title=doc.title,
            )
        )
        # Prepare overlap
        if overlap_chars > 0:
            keep_chars = min(overlap_chars, len(text))
            keep_text = text[-keep_chars:]
            # Map overlap to the last page of previous chunk for simplicity
            buffer = [(end_page_inclusive, keep_text)]
            buffer_char_len = len(keep_text)
            buffer_start_page = end_page_inclusive
        else:
            buffer = []
            buffer_char_len = 0

    for i, page in enumerate(doc.pages):
        content = page.text
        if not content:
            continue
        # Ensure page separation to avoid accidental merges
        page_text = content.rstrip() + "\n\n"
        if buffer_char_len == 0:
            buffer_start_page = i
        buffer.append((i, page_text))
        buffer_char_len += len(page_text)
        if buffer_char_len >= chunk_chars:
            flush(i)

    # Flush remainder
    if buffer:
        flush(doc.pages[len(doc.pages) - 1].file_page_index if doc.pages else 0)

    return chunks


class YCEmbedder:
    """
    Клиент для получения embeddings через Yandex Cloud Foundation Models.
    Attributes:
        api_key (Optional[str]): API ключ
        iam_token (Optional[str]): IAM токен
        folder_id (Optional[str]): ID папки
        variant (str): Вариант embeddings
        endpoint (str): URL endpoint
        timeout (int): Таймаут запроса
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        iam_token: Optional[str] = None,
        folder_id: Optional[str] = None,
        variant: str = "text-search-doc",
        endpoint: str = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding",
        timeout: int = 30,
    ) -> None:
        if not api_key and not iam_token:
            raise ValueError("Provide either api_key or iam_token for Yandex Cloud.")
        if not folder_id:
            raise ValueError("folder_id is required for Yandex Cloud embeddings.")
        self.api_key = api_key
        self.iam_token = iam_token
        self.folder_id = folder_id
        self.variant = variant
        self.endpoint = endpoint
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        """Создает заголовки для запроса."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Api-Key {self.api_key}"
        elif self.iam_token:
            headers["Authorization"] = f"Bearer {self.iam_token}"
        return headers

    def _post_text(self, model_uri: str, text: str) -> List[float]:
        """
        Отправляет текст для получения embeddings.
        Args:
            model_uri (str): URI модели
            text (str): Текст для обработки
        Returns:
            List[float]: Вектор embeddings
        Raises:
            RuntimeError: При ошибке API
        """
        payload = {
            "modelUri": model_uri,
            "text": text,
        }
        resp = requests.post(
            self.endpoint,
            headers=self._headers(),
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"YC embeddings error: {resp.status_code} {resp.text}")
        data = resp.json()
        vec = None
        if isinstance(data, dict):
            emb = data.get("embedding")
            if isinstance(emb, dict) and "vector" in emb:
                vec = emb["vector"]
            elif isinstance(emb, list):
                vec = emb
            elif "vector" in data:
                vec = data["vector"]
        if vec is None or not isinstance(vec, list):
            raise RuntimeError("Unexpected YC embeddings response format.")
        return [float(x) for x in vec]

    @staticmethod
    def _split_text(text: str) -> Tuple[str, str]:
        """
        Разделяет текст на две части.
        Args:
            text (str): Текст для разделения
        Returns:
            Tuple[str, str]: Две части текста
        """
        mid = len(text) // 2
        # Prefer splitting on paragraph boundary near middle
        left_break = text.rfind("\n\n", 0, mid)
        right_break = text.find("\n\n", mid)
        if left_break != -1 and (right_break == -1 or mid - left_break <= right_break - mid):
            split_at = left_break + 2
        elif right_break != -1:
            split_at = right_break + 2
        else:
            # Fallback: split at whitespace near middle
            left_space = text.rfind(" ", 0, mid)
            right_space = text.find(" ", mid)
            if left_space != -1 and (right_space == -1 or mid - left_space <= right_space - mid):
                split_at = left_space + 1
            elif right_space != -1:
                split_at = right_space + 1
            else:
                split_at = mid
        return text[:split_at].strip(), text[split_at:].strip()

    def _embed_with_split(self, model_uri: str, text: str, depth: int = 0, max_depth: int = 6) -> List[float]:
        """
        Получает embeddings с возможностью рекурсивного разделения текста.
        Args:
            model_uri (str): URI модели
            text (str): Текст для обработки
            depth (int): Текущая глубина рекурсии
            max_depth (int): Максимальная глубина рекурсии
        Returns:
            List[float]: Вектор embeddings
        Raises:
            RuntimeError: При ошибке API или превышении глубины рекурсии
        """
        try:
            return self._post_text(model_uri, text)
        except RuntimeError as e:
            msg = str(e)
            # Detect token limit error explicitly
            if "number of input tokens must be no more than" in msg and depth < max_depth and len(text) > 0:
                left, right = self._split_text(text)
                # If split fails to reduce size, rethrow
                if not left or not right:
                    raise
                v1 = self._embed_with_split(model_uri, left, depth + 1, max_depth)
                v2 = self._embed_with_split(model_uri, right, depth + 1, max_depth)
                # Average vectors
                a = np.array(v1, dtype=np.float32)
                b = np.array(v2, dtype=np.float32)
                avg = ((a + b) / 2.0).astype(np.float32)
                return avg.tolist()
            raise

    def embed_texts(self, texts: List[str], batch_size: int = 32, verbose: bool = False) -> np.ndarray:
        """
        Получает embeddings для списка текстов.
        Args:
            texts (List[str]): Список текстов
            batch_size (int): Размер батча
            verbose (bool): Показывать ли прогресс-бар
        Returns:
            np.ndarray: Массив embeddings
        Raises:
            RuntimeError: При несоответствии количества текстов и embeddings
        """
        vectors: List[List[float]] = []
        model_uri = f"emb://{self.folder_id}/{self.variant}/latest"
        # The API commonly accepts only single 'text' per request. We loop.
        iterator = range(0, len(texts), batch_size)
        if verbose:
            iterator = tqdm(
                iterator,
                total=(len(texts) + batch_size - 1) // batch_size,
                desc="Embedding",
            )
        for i in iterator:
            batch = texts[i : i + batch_size]
            inner_iter = batch
            if verbose:
                inner_iter = tqdm(inner_iter, leave=False, desc="Requests")
            for text in inner_iter:
                vec = self._embed_with_split(model_uri, text)
                vectors.append(vec)
        arr = np.array(vectors, dtype=np.float32)
        if arr.shape[0] != len(texts):
            raise RuntimeError("Mismatch between number of texts and embeddings returned.")
        return arr


def save_index(output_dir: str, chunks: List[Chunk], embeddings: np.ndarray) -> None:
    """
    Сохраняет индекс в файлы.
    Args:
        output_dir (str): Директория для сохранения
        chunks (List[Chunk]): Список чанков
        embeddings (np.ndarray): Массив embeddings
    """
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings.astype(np.float32))
    meta_path = os.path.join(output_dir, "meta.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for idx, ch in enumerate(chunks):
            record = {
                "id": idx,
                "file_path": ch.file_path,
                "author": ch.author,
                "year": ch.year,
                "title": ch.title,
                "chapter_titles": ch.chapter_titles,
                "page_start_label": ch.page_start_label,
                "page_end_label": ch.page_end_label,
                "file_page_start": ch.page_start_index + 1,
                "file_page_end": ch.page_end_index + 1,
                "chunk": ch.text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _compute_chunks_signature(chunks: List[Chunk]) -> str:
    """
    Вычисляет SHA256 хеш для проверки неизменности чанков.
    Args:
        chunks (List[Chunk]): Список чанков
    Returns:
        str: SHA256 хеш
    """
    hasher = hashlib.sha256()
    for ch in chunks:
        # Use only stable fields affecting retrieval; ignore ids
        hasher.update((ch.file_path or "").encode("utf-8"))
        hasher.update((ch.author or "").encode("utf-8"))
        hasher.update((str(ch.year) if ch.year is not None else "").encode("utf-8"))
        hasher.update((ch.title or "").encode("utf-8"))
        hasher.update("|".join(ch.chapter_titles).encode("utf-8"))
        hasher.update(str(ch.page_start_index).encode("utf-8"))
        hasher.update(str(ch.page_end_index).encode("utf-8"))
        # Include text content; to keep it efficient, hash of text
        hasher.update(hashlib.sha256(ch.text.encode("utf-8")).digest())
    return hasher.hexdigest()


def _manifest_paths(output_dir: str) -> Tuple[str, str, str]:
    """
    Возвращает пути к файлам манифеста.
    Args:
        output_dir (str): Директория с индексом
    Returns:
        Tuple[str, str, str]: Пути к meta.jsonl, embeddings.npy, manifest.json
    """
    return (
        os.path.join(output_dir, "meta.jsonl"),
        os.path.join(output_dir, "embeddings.npy"),
        os.path.join(output_dir, "manifest.json"),
    )


def _load_manifest(manifest_path: str) -> Optional[Dict]:
    """
    Загружает манифест из файла.
    Args:
        manifest_path (str): Путь к файлу манифеста
    Returns:
        Optional[Dict]: Содержимое манифеста или None
    """
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_manifest(manifest_path: str, manifest: Dict) -> None:
    """
    Сохраняет манифест в файл.
    Args:
        manifest_path (str): Путь к файлу манифеста
        manifest (Dict): Данные манифеста
    """
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, manifest_path)


def build_index(
    input_dir: str,
    output_dir: str,
    yc_api_key: Optional[str],
    yc_iam_token: Optional[str],
    yc_folder_id: str,
    yc_variant: str,
    yc_endpoint: str,
    chunk_chars: int,
    overlap_chars: int,
    batch_size: int,
    verbose: bool = False,
) -> None:
    """
    Строит индекс из документов в указанной директории.
    Args:
        input_dir (str): Директория с документами
        output_dir (str): Директория для сохранения индекса
        yc_api_key (Optional[str]): API ключ Yandex Cloud
        yc_iam_token (Optional[str]): IAM токен Yandex Cloud
        yc_folder_id (str): ID папки Yandex Cloud
        yc_variant (str): Вариант embeddings
        yc_endpoint (str): URL endpoint API
        chunk_chars (int): Размер чанка в символах
        overlap_chars (int): Размер перекрытия между чанками
        batch_size (int): Размер батча для embeddings
        verbose (bool): Показывать ли прогресс-бар
    """
    # Collect all PDFs and DJVUs
    pdf_files: List[str] = []
    djvu_files: List[str] = []
    for root, _dirs, files in os.walk(input_dir):
        for fn in files:
            if fn.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, fn))
            elif fn.lower().endswith((".djvu", ".djv")):
                djvu_files.append(os.path.join(root, fn))
    if not pdf_files and not djvu_files:
        print(f"No PDF or DJVU files found in {input_dir}")
        return

    all_chunks: List[Chunk] = []
    # Parse PDFs
    for path in sorted(pdf_files):
        try:
            doc = parse_pdf(path)
        except Exception as e:
            print(f"Failed to parse {path}: {e}")
            continue
        doc_chunks = chunk_document(doc, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
        doc_chunks = [c for c in doc_chunks if c.text and c.text.strip()]
        print(f"Parsed PDF {os.path.basename(path)}: pages={len(doc.pages)}, chunks={len(doc_chunks)}")
        all_chunks.extend(doc_chunks)
    # Parse DJVUs
    for path in sorted(djvu_files):
        try:
            doc = parse_djvu(path)
        except Exception as e:
            print(f"Failed to parse {path}: {e}")
            continue
        doc_chunks = chunk_document(doc, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
        doc_chunks = [c for c in doc_chunks if c.text and c.text.strip()]
        print(f"Parsed DJVU {os.path.basename(path)}: pages={len(doc.pages)}, chunks={len(doc_chunks)}")
        all_chunks.extend(doc_chunks)

    if not all_chunks:
        print("No chunks to embed.")
        return

    # Idempotency: if existing index matches current chunks signature, skip embeddings
    meta_path, emb_path, manifest_path = _manifest_paths(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    chunks_sig = _compute_chunks_signature(all_chunks)
    old_manifest = _load_manifest(manifest_path)
    if (
        old_manifest
        and old_manifest.get("chunks_signature") == chunks_sig
        and os.path.exists(meta_path)
        and os.path.exists(emb_path)
    ):
        print("Index is up-to-date. Skipping embeddings.")
        return

    embedder = YCEmbedder(
        api_key=yc_api_key,
        iam_token=yc_iam_token,
        folder_id=yc_folder_id,
        variant=yc_variant,
        endpoint=yc_endpoint,
    )
    texts = [c.text for c in all_chunks]
    embeddings = embedder.embed_texts(texts, batch_size=batch_size, verbose=verbose)
    print(f"Embeddings computed: shape={embeddings.shape}")
    save_index(output_dir, all_chunks, embeddings)
    print(f"Saved index to {output_dir}/embeddings.npy and {output_dir}/meta.jsonl")
    _save_manifest(
        manifest_path,
        {
            "chunks_signature": chunks_sig,
            "num_chunks": len(all_chunks),
            "yc_variant": yc_variant,
        },
    )


def main(argv: Optional[List[str]] = None) -> int:
    """
    Основная функция для запуска из командной строки.
    Args:
        argv (Optional[List[str]]): Аргументы командной строки
    Returns:
        int: Код возврата
    """
    parser = argparse.ArgumentParser(
        description="Build a local RAG index (embeddings.npy + meta.jsonl) from PDFs and DJVUs."
    )
    parser.add_argument("--input_dir", type=str, default="data", help="Directory with input PDFs/DJVUs")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="index",
        help="Directory to store embeddings and metadata",
    )
    parser.add_argument("--chunk_chars", type=int, default=2000, help="Approximate characters per chunk")
    parser.add_argument(
        "--overlap_chars",
        type=int,
        default=200,
        help="Overlap characters between chunks",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for YC embeddings API")
    parser.add_argument("--verbose", action="store_true", help="Show progress bar for embeddings")
    # Yandex Cloud config
    parser.add_argument(
        "--yc_api_key",
        type=str,
        default=os.getenv("YC_API_KEY"),
        help="Yandex Cloud API Key (env YC_API_KEY)",
    )
    parser.add_argument(
        "--yc_iam_token",
        type=str,
        default=os.getenv("YC_IAM_TOKEN"),
        help="Yandex Cloud IAM token (env YC_IAM_TOKEN)",
    )
    parser.add_argument(
        "--yc_folder_id",
        type=str,
        default=os.getenv("YC_FOLDER_ID"),
        help="Yandex Cloud Folder ID (env YC_FOLDER_ID)",
    )
    parser.add_argument(
        "--yc_variant",
        type=str,
        default=os.getenv("YC_EMBEDDING_VARIANT", "text-search-doc"),
        help="YC embedding variant: text-search-doc or text-search-query",
    )
    parser.add_argument(
        "--yc_endpoint",
        type=str,
        default=os.getenv(
            "YC_EMBEDDING_ENDPOINT",
            "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding",
        ),
        help="YC embeddings endpoint",
    )

    args = parser.parse_args(argv)

    if not args.yc_folder_id:
        print("Error: --yc_folder_id or env YC_FOLDER_ID is required", file=sys.stderr)
        return 2
    if not args.yc_api_key and not args.yc_iam_token:
        print(
            "Error: Provide --yc_api_key or --yc_iam_token (or set env YC_API_KEY/YC_IAM_TOKEN)",
            file=sys.stderr,
        )
        return 2

    try:
        build_index(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            yc_api_key=args.yc_api_key,
            yc_iam_token=args.yc_iam_token,
            yc_folder_id=args.yc_folder_id,
            yc_variant=args.yc_variant,
            yc_endpoint=args.yc_endpoint,
            chunk_chars=args.chunk_chars,
            overlap_chars=args.overlap_chars,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"Failed to build index: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
