"""E-book loading and text segmentation.

Supported formats: EPUB, FB2, TXT, HTML, Markdown (plain read).
The book is split into "passages" - sentence groups of manageable length
that are analyzed & synthesized one by one.
"""
from __future__ import annotations

import hashlib
import os
import re
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser


@dataclass
class Passage:
    index: int
    text: str
    chapter: str = ""


@dataclass
class Book:
    id: str
    title: str
    author: str
    language: str
    passages: list[Passage] = field(default_factory=list)
    path: str = ""


# ---------------------------------------------------------------------------
# HTML / XML text extraction
# ---------------------------------------------------------------------------
class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "head"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        if tag in ("p", "br", "div", "h1", "h2", "h3", "h4", "li", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str | bytes) -> str:
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="ignore")
    p = _TextExtractor()
    try:
        p.feed(html)
    except Exception:
        pass
    text = "".join(p.parts)
    text = re.sub(r"[ \t\r]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_epub(path: str) -> tuple[str, str, str, list[str]]:
    """Return (title, author, language, [chapter texts])."""
    import ebooklib
    from ebooklib import epub
    book = epub.read_epub(path, options={"ignore_ncx": True})
    title = next(iter(book.get_metadata("DC", "title")), ["Untitled"])[0]
    author = next(iter(book.get_metadata("DC", "creator")), [""])[0]
    lang = next(iter(book.get_metadata("DC", "language")), ["en"])[0]
    chapters = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        t = html_to_text(item.get_content())
        # drop near-empty nav pages
        if len(re.sub(r"\W", "", t)) > 40:
            chapters.append(t)
    return title, author, lang, chapters


def parse_fb2(path: str) -> tuple[str, str, str, list[str]]:
    from lxml import etree
    ns = {"f": "http://www.gribuser.ru/xml/fictionbook/2.0"}
    tree = etree.parse(path)
    root = tree.getroot()

    def txt(el_list):
        return " ".join("".join(e.itertext()).strip() for e in el_list).strip()

    title = txt(root.findall(".//f:book-title", ns)) or "Untitled"
    author = ", ".join(filter(None, [
        txt(root.findall(".//f:first-name", ns)),
        txt(root.findall(".//f:last-name", ns))]))
    lang = txt(root.findall(".//f:lang", ns)) or "ru"
    chapters = []
    for ch in root.findall(".//f:body/f:section", ns):
        name = txt(ch.findall("./f:title", ns))
        body = html_to_text(etree.tostring(ch))
        if len(re.sub(r"\W", "", body)) > 40:
            chapters.append((name + "\n" if name else "") + body)
    return title, author, lang, chapters


def parse_txt(path: str) -> tuple[str, str, str, list[str]]:
    raw = open(path, "rb").read()
    for enc in ("utf-8", "cp1251", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="ignore")
    # naive CJK detection
    lang = "ru" if re.search(r"[а-яА-ЯёЁ]", text[:4000]) else "en"
    # split on empty lines / chapter headings
    parts = re.split(r"\n\s*\n(?=\s*(?:Глава|Chapter|CHAPTER|Part|#|\d+\.)|\s*$)", text)
    parts = [p.strip() for p in parts if len(re.sub(r"\W", "", p)) > 40]
    title = os.path.splitext(os.path.basename(path))[0]
    first_line = text.strip().splitlines()[0][:80] if text.strip() else title
    return first_line or title, "", lang, parts or [text]


# ---------------------------------------------------------------------------
# Segmentation into passages
# ---------------------------------------------------------------------------
SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[\"“»A-ZА-ЯЁa-zа-яё])")


def segment(chapters: list[str], max_words: int = 45) -> list[Passage]:
    passages: list[Passage] = []
    idx = 0
    for ch_no, ch in enumerate(chapters, 1):
        lines = [l.strip() for l in ch.split("\n") if l.strip()]
        chapter_name = lines[0][:60] if lines and len(lines[0]) < 70 else f"Глава {ch_no}"
        paragraph = " ".join(l for l in lines[1:] if not l.endswith(":")) \
            if len(lines) > 1 else " ".join(lines)
        sentences = SENT_SPLIT.split(paragraph)
        buf: list[str] = []
        words = 0
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            w = len(s.split())
            # very long sentence: hard-split by commas
            while w > max_words * 1.6:
                cut = s.rfind(",", 0, int(len(s) * 0.6))
                if cut < 40:
                    cut = len(s) // 2
                buf.append(s[:cut + 1])
                passages.append(Passage(idx, " ".join(buf), chapter_name))
                idx += 1
                buf, words = [], 0
                s = s[cut + 1:].strip()
                w = len(s.split())
            buf.append(s)
            words += w
            if words >= max_words:
                passages.append(Passage(idx, " ".join(buf), chapter_name))
                idx += 1
                buf, words = [], 0
        if buf:
            passages.append(Passage(idx, " ".join(buf), chapter_name))
            idx += 1
    return passages


def load_book(path: str, max_passages: int | None = None) -> Book:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".epub":
        title, author, lang, chapters = parse_epub(path)
    elif ext in (".fb2", ".xml"):
        title, author, lang, chapters = parse_fb2(path)
    elif ext in (".txt", ".md"):
        title, author, lang, chapters = parse_txt(path)
    elif ext in (".html", ".htm", ".xhtml"):
        t = html_to_text(open(path, "rb").read())
        title, lang = os.path.basename(path), ("ru" if re.search(r"[а-я]", t[:3000]) else "en")
        chapters = [t]
        author = ""
    else:
        raise ValueError(f"Неподдерживаемый формат: {ext}")

    passages = segment(chapters)
    if max_passages:
        passages = passages[:max_passages]
    bid = hashlib.sha1((title + str(len(passages))).encode()).hexdigest()[:12]
    return Book(id=bid, title=title or "Untitled", author=author,
                language=lang or "en", passages=passages, path=path)
