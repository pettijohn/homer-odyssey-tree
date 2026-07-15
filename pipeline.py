#!/usr/bin/env python3
"""Download, parse, summarize, and build a tree viewer for The Odyssey."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias, cast

from bs4 import BeautifulSoup, Tag


ROOT: Final[Path] = Path(__file__).resolve().parent
SOURCE_URL: Final[str] = "https://www.gutenberg.org/cache/epub/1727/pg1727-images.html"
SOURCE_PATH: Final[Path] = ROOT / "source" / "pg1727-images.html"
PARSED_PATH: Final[Path] = ROOT / "data" / "parsed.json"
CACHE_PATH: Final[Path] = ROOT / "data" / "summary-cache.json"
SUMMARY_PATH: Final[Path] = ROOT / "data" / "odyssey.json"
TEMPLATE_PATH: Final[Path] = ROOT / "viewer_template.html"
OUTPUT_PATH: Final[Path] = ROOT / "index.html"
MODEL: Final[str] = "gemma-4-26B-A4B"
SENTENCE_BOUNDARY: Final[re.Pattern[str]] = re.compile(
    r"[.!?][”\"]?\s+(?=[A-Z“\"])")
ABBREVIATION: Final[re.Pattern[str]] = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Mt|St|Jr|Sr|vs|etc)\.", re.IGNORECASE
)
LEADING_INTERJECTION: Final[re.Pattern[str]] = re.compile(
    r"^(?P<quote>[“\"]?)(?:Alas|Ah|Oh)!\s+", re.IGNORECASE
)
SummaryKind: TypeAlias = Literal["sentence", "paragraph"]


@dataclass(frozen=True)
class SourceParagraph:
    number: int
    source: str


@dataclass(frozen=True)
class SourceBook:
    number: int
    title: str
    paragraphs: list[SourceParagraph]


@dataclass(frozen=True)
class SummarizedParagraph:
    number: int
    summary: str
    source: str


@dataclass(frozen=True)
class SummarizedBook:
    number: int
    title: str
    summary: str
    paragraphs: list[SummarizedParagraph]


class SummaryCache:
    """A thread-safe, eagerly persisted summary cache."""

    def __init__(self, path: Path) -> None:
        self.path: Path = path
        self.lock: threading.Lock = threading.Lock()
        self.values: dict[str, str] = self._load()

    def _load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        raw: object = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in raw.items()
        ):
            raise ValueError(f"Invalid summary cache: {self.path}")
        return cast(dict[str, str], raw)

    def get(self, key: str) -> str | None:
        with self.lock:
            return self.values.get(key)

    def put(self, key: str, value: str) -> None:
        with self.lock:
            self.values[key] = value
            write_json(self.path, self.values)


def write_json(path: Path, value: object) -> None:
    """Atomically write readable UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int
    temporary_name: str
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path: Path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def download_source(force: bool = False) -> Path:
    """Download Gutenberg's HTML edition, retaining it in the repository."""
    if SOURCE_PATH.exists() and not force:
        return SOURCE_PATH
    SOURCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    request: urllib.request.Request = urllib.request.Request(
        SOURCE_URL, headers={"User-Agent": "odyssey-tree-builder/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload: bytes = response.read()
    if len(payload) < 100_000 or b'id="chap01"' not in payload:
        raise RuntimeError("The Gutenberg download did not look like the expected book HTML")
    SOURCE_PATH.write_bytes(payload)
    return SOURCE_PATH


def clean_text(tag: Tag) -> str:
    """Collapse Gutenberg's presentation whitespace without altering words."""
    return " ".join(tag.get_text(" ", strip=True).split())


def parse_source(path: Path) -> list[SourceBook]:
    """Extract paragraph text strictly from Books I through XXIV."""
    soup: BeautifulSoup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    books: list[SourceBook] = []
    for book_number in range(1, 25):
        anchor: Tag | None = soup.find(id=f"chap{book_number:02d}")
        if anchor is None or not isinstance(anchor.parent, Tag) or anchor.parent.name != "h2":
            raise RuntimeError(f"Could not find heading for book {book_number}")
        heading: Tag = anchor.parent
        paragraphs: list[SourceParagraph] = []
        for sibling in heading.next_siblings:
            if isinstance(sibling, Tag) and sibling.name == "h2":
                break
            if isinstance(sibling, Tag) and sibling.name == "p":
                text: str = clean_text(sibling)
                if text:
                    paragraphs.append(SourceParagraph(number=len(paragraphs) + 1, source=text))
        if not paragraphs:
            raise RuntimeError(f"Book {book_number} contained no paragraphs")
        books.append(SourceBook(book_number, clean_text(heading), paragraphs))
    if len(books) != 24:
        raise RuntimeError(f"Expected 24 books, found {len(books)}")
    return books


def prompt_for(text: str, kind: SummaryKind) -> str:
    """Create the exact local-model instruction required by the brief."""
    return (
        f"Summarize this text into one {kind}. DO NOT include any preamble, ONLY respond "
        f"with the summary. Preserve the source style or tone: '{text}'"
    )


def cache_key(text: str, kind: SummaryKind) -> str:
    """Key summaries by all inputs that affect model output."""
    material: str = f"{MODEL}\0{kind}\0{text}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def remove_leading_interjection(match: re.Match[str]) -> str:
    """Retain an opening quote while removing an interjection boundary."""
    return match.group("quote")


def hide_abbreviation_period(match: re.Match[str]) -> str:
    """Prevent a period in a known abbreviation from looking like a boundary."""
    return match.group(0).replace(".", "․")


def has_requested_shape(summary: str, kind: SummaryKind) -> bool:
    """Reject preambles, multiple paragraphs, and multi-sentence sentence outputs."""
    if not summary or summary.strip() != summary or "\n\n" in summary:
        return False
    lowered: str = summary.lower()
    if lowered.startswith(("here is", "summary:", "the text")):
        return False
    if kind == "paragraph":
        return True
    without_interjection: str = LEADING_INTERJECTION.sub(remove_leading_interjection, summary)
    without_abbreviations: str = ABBREVIATION.sub(hide_abbreviation_period, without_interjection)
    return SENTENCE_BOUNDARY.search(without_abbreviations) is None


def invoke_model(text: str, kind: SummaryKind, retries: int = 3) -> str:
    """Invoke the mandated bun/pi model command and return its plain output."""
    last_error: str = "unknown failure"
    candidate_text: str = text
    for attempt in range(1, retries + 1):
        command: list[str] = [
            "bun",
            "run",
            "pi",
            "-p",
            "--model",
            MODEL,
            "--thinking",
            "off",
            prompt_for(candidate_text, kind),
        ]
        process: subprocess.CompletedProcess[str] = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=300,
            check=False,
        )
        summary: str = process.stdout.strip()
        if process.returncode == 0 and has_requested_shape(summary, kind):
            return summary
        if process.returncode == 0 and summary:
            # Keep correction model-driven: summarize the nonconforming model output
            # with the same mandated prompt on the next attempt.
            candidate_text = summary
        last_error = process.stderr.strip() or (
            f"output did not have the requested {kind} shape (exit {process.returncode})"
        )
        print(f"Model attempt {attempt}/{retries} failed: {last_error}", file=sys.stderr)
    raise RuntimeError(f"Model summarization failed after {retries} attempts: {last_error}")


def summarize_cached(text: str, kind: SummaryKind, cache: SummaryCache) -> str:
    """Reuse a prior result or generate and persist a new one."""
    key: str = cache_key(text, kind)
    existing: str | None = cache.get(key)
    if existing is not None and has_requested_shape(existing, kind):
        return existing
    summary: str = invoke_model(text, kind)
    cache.put(key, summary)
    return summary


def validate_prompt(books: list[SourceBook]) -> None:
    """Exercise the prompt on short, medium, and book-length source samples."""
    samples: list[tuple[str, SummaryKind, str]] = [
        (books[0].paragraphs[1].source, "sentence", "short paragraph"),
        (books[3].paragraphs[12].source, "sentence", "dialogue paragraph"),
        ("\n\n".join(p.source for p in books[5].paragraphs), "paragraph", "complete book"),
    ]
    print("Validating the mandated prompt with three samples:")
    for text, kind, label in samples:
        summary: str = invoke_model(text, kind)
        print(f"[{label} -> {kind}] {summary}")


def summarize_books(books: list[SourceBook], workers: int) -> list[SummarizedBook]:
    """Generate cached paragraph and book summaries with bounded concurrency."""
    cache: SummaryCache = SummaryCache(CACHE_PATH)
    tasks: list[tuple[int, int, str]] = [
        (book.number, paragraph.number, paragraph.source)
        for book in books
        for paragraph in book.paragraphs
    ]
    paragraph_summaries: dict[tuple[int, int], str] = {}

    def do_paragraph(task: tuple[int, int, str]) -> tuple[int, int, str]:
        book_number, paragraph_number, source = task
        summary: str = summarize_cached(source, "sentence", cache)
        return book_number, paragraph_number, summary

    total: int = len(tasks)
    completed: int = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures: list[concurrent.futures.Future[tuple[int, int, str]]] = [
            executor.submit(do_paragraph, task) for task in tasks
        ]
        for future in concurrent.futures.as_completed(futures):
            book_number, paragraph_number, summary = future.result()
            paragraph_summaries[(book_number, paragraph_number)] = summary
            completed += 1
            if completed == 1 or completed % 25 == 0 or completed == total:
                print(f"Paragraph summaries: {completed}/{total}", flush=True)

    summarized: list[SummarizedBook] = []
    for book in books:
        source_text: str = "\n\n".join(paragraph.source for paragraph in book.paragraphs)
        print(f"Summarizing {book.title}...", flush=True)
        book_summary: str = summarize_cached(source_text, "paragraph", cache)
        summarized_paragraphs: list[SummarizedParagraph] = [
            SummarizedParagraph(
                number=paragraph.number,
                summary=paragraph_summaries[(book.number, paragraph.number)],
                source=paragraph.source,
            )
            for paragraph in book.paragraphs
        ]
        summarized.append(SummarizedBook(book.number, book.title, book_summary, summarized_paragraphs))
    return summarized


def build_html(data: dict[str, Any]) -> None:
    """Embed all results in a single dependency-free HTML document."""
    template: str = TEMPLATE_PATH.read_text(encoding="utf-8")
    marker: str = "__ODYSSEY_DATA__"
    if template.count(marker) != 1:
        raise RuntimeError(f"Expected exactly one {marker} marker in {TEMPLATE_PATH}")
    embedded: str = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    OUTPUT_PATH.write_text(template.replace(marker, embedded), encoding="utf-8")


def load_summary_data() -> dict[str, Any]:
    """Load the generated summary artifact for a viewer-only rebuild."""
    raw: object = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid summary data: {SUMMARY_PATH}")
    return cast(dict[str, Any], raw)


def verify_artifacts(source_books: list[SourceBook], data: dict[str, Any]) -> None:
    """Verify hierarchy, source fidelity, summary shape, and standalone output."""
    raw_books: object = data.get("books")
    if not isinstance(raw_books, list) or len(raw_books) != 24:
        raise RuntimeError("Summary data must contain exactly 24 books")
    books: list[dict[str, Any]] = cast(list[dict[str, Any]], raw_books)
    if [book.get("number") for book in books] != list(range(1, 25)):
        raise RuntimeError("Summary books are missing or out of order")
    summary_count: int = 0
    for source_book, book in zip(source_books, books, strict=True):
        book_summary: object = book.get("summary")
        if not isinstance(book_summary, str) or not has_requested_shape(book_summary, "paragraph"):
            raise RuntimeError(f"{source_book.title} does not have a one-paragraph summary")
        raw_paragraphs: object = book.get("paragraphs")
        if not isinstance(raw_paragraphs, list) or len(raw_paragraphs) != len(source_book.paragraphs):
            raise RuntimeError(f"Paragraph count mismatch in {source_book.title}")
        paragraphs: list[dict[str, Any]] = cast(list[dict[str, Any]], raw_paragraphs)
        for source_paragraph, paragraph in zip(source_book.paragraphs, paragraphs, strict=True):
            if paragraph.get("source") != source_paragraph.source:
                raise RuntimeError(
                    f"Source mismatch in {source_book.title}, paragraph {source_paragraph.number}"
                )
            summary: object = paragraph.get("summary")
            if not isinstance(summary, str) or not has_requested_shape(summary, "sentence"):
                raise RuntimeError(
                    f"Invalid sentence summary in {source_book.title}, paragraph {source_paragraph.number}"
                )
            summary_count += 1
    if summary_count != 1051:
        raise RuntimeError(f"Expected 1051 paragraph nodes, found {summary_count}")
    html: str = OUTPUT_PATH.read_text(encoding="utf-8")
    if "__ODYSSEY_DATA__" in html or html.count("const ODYSSEY_DATA = ") != 1:
        raise RuntimeError("Standalone HTML does not contain exactly one embedded data set")
    if "<script src=" in html or "<link rel=\"stylesheet\"" in html:
        raise RuntimeError("Standalone HTML unexpectedly depends on an external script or stylesheet")
    print("Verified 24 books, 1051 sentence/source nodes, summary shapes, and standalone HTML.")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("all", "download", "parse", "validate", "summarize", "build", "verify"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--workers", type=int, default=4, help="parallel model calls (default: 4)")
    parser.add_argument("--force-download", action="store_true", help="replace the saved Gutenberg HTML")
    parser.add_argument("--skip-validation", action="store_true", help="skip three prompt checks in the all command")
    return parser.parse_args()


def main() -> None:
    """Run the selected reproducible pipeline stage."""
    args: argparse.Namespace = parse_arguments()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.command == "download":
        print(download_source(args.force_download))
        return

    source_path: Path = download_source(args.force_download)
    books: list[SourceBook] = parse_source(source_path)
    parsed_data: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "books": [asdict(book) for book in books],
    }
    write_json(PARSED_PATH, parsed_data)
    print(f"Parsed {sum(len(book.paragraphs) for book in books)} paragraphs across 24 books.")
    if args.command == "parse":
        return
    if args.command == "validate":
        validate_prompt(books)
        return
    if args.command == "build":
        build_html(load_summary_data())
        print(f"Built {OUTPUT_PATH}")
        return
    if args.command == "verify":
        verify_artifacts(books, load_summary_data())
        return
    if args.command == "all" and not args.skip_validation:
        validate_prompt(books)

    summarized: list[SummarizedBook] = summarize_books(books, args.workers)
    final_data: dict[str, Any] = {
        "title": "The Odyssey",
        "source_url": SOURCE_URL,
        "model": MODEL,
        "books": [asdict(book) for book in summarized],
    }
    write_json(SUMMARY_PATH, final_data)
    build_html(final_data)
    verify_artifacts(books, final_data)
    print(f"Wrote {SUMMARY_PATH} and {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
