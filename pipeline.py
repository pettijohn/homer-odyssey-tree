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
SOURCE_EDITION: Final[str] = "pg1728"
SOURCE_URL: Final[str] = "https://www.gutenberg.org/cache/epub/1728/pg1728-images.html"
SOURCE_PATH: Final[Path] = ROOT / "source" / "pg1728-images.html"
PARSED_PATH: Final[Path] = ROOT / "data" / "parsed.json"
CACHE_PATH: Final[Path] = ROOT / "data" / "summary-cache.json"
SUMMARY_PATH: Final[Path] = ROOT / "data" / "odyssey.json"
TEMPLATE_PATH: Final[Path] = ROOT / "viewer_template.html"
OUTPUT_PATH: Final[Path] = ROOT / "index.html"
MODEL: Final[str] = "gemma-4-26B-A4B"
OUTLINE_MIN_PARAGRAPHS: Final[int] = 5
OUTLINE_MAX_PARAGRAPHS: Final[int] = 10
OUTLINE_PROMPT_VERSION: Final[str] = "range-outline-v7-firm-target-thinking"
SENTENCE_BOUNDARY: Final[re.Pattern[str]] = re.compile(
    r"[.!?][”\"]?\s+(?=[A-Z“\"])")
ABBREVIATION: Final[re.Pattern[str]] = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Mt|St|Jr|Sr|vs|etc)\.", re.IGNORECASE
)
LEADING_INTERJECTION: Final[re.Pattern[str]] = re.compile(
    r"^(?P<quote>[“\"]?)(?:Alas|Ah|Lo|Oh)!\s+", re.IGNORECASE
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
class OutlineRange:
    summary: str
    start_paragraph: int
    end_paragraph: int


@dataclass(frozen=True)
class SummarizedSection:
    number: int
    summary: str
    start_paragraph: int
    end_paragraph: int
    paragraphs: list[SummarizedParagraph]


@dataclass(frozen=True)
class SummarizedBook:
    number: int
    title: str
    summary: str
    sections: list[SummarizedSection]


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
                classes: object = sibling.get("class", [])
                if isinstance(classes, list) and "footnote" in classes:
                    continue
                if sibling.find(id=re.compile(r"^linknote-\d+$")) is not None:
                    continue
                text: str = clean_text(sibling)
                if text:
                    paragraphs.append(SourceParagraph(number=len(paragraphs) + 1, source=text))
        if not paragraphs:
            raise RuntimeError(f"Book {book_number} contained no paragraphs")
        books.append(SourceBook(book_number, clean_text(heading).rstrip("."), paragraphs))
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
    material: str = f"{SOURCE_EDITION}\0{MODEL}\0{kind}\0{text}"
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


def invoke_model(text: str, kind: SummaryKind, retries: int = 6) -> str:
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
            correction: str = (
                "A draft summary follows. Rewrite it as exactly one grammatical sentence, "
                "using indirect speech instead of sentence-ending punctuation inside quoted "
                "dialogue when necessary.\n\n"
                if kind == "sentence"
                else "A draft summary follows. Rewrite it as exactly one prose paragraph.\n\n"
            )
            candidate_text = correction + summary
        last_error = process.stderr.strip() or (
            f"output did not have the requested {kind} shape "
            f"(exit {process.returncode}): {summary[:500]!r}"
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


def outline_prompt(book_title: str, paragraphs: list[SummarizedParagraph]) -> str:
    """Request a flat, strictly verifiable hierarchy over cached paragraph summaries."""
    inputs: list[dict[str, int | str]] = [
        {"paragraph_number": paragraph.number, "summary_sentence": paragraph.summary}
        for paragraph in paragraphs
    ]
    paragraph_count: int = len(paragraphs)
    return f"""Organize these existing paragraph summaries from one book of The Odyssey into a hierarchical outline.

Return exactly one valid JSON array and nothing else. Do not use Markdown fences, commentary, or a wrapper object.

Every array item must use exactly this shape:
{{"summary_sentence":"Exactly one grammatical sentence summarizing this group.","start_paragraph":1,"end_paragraph":5}}

Rules:
- Partition the input into contiguous ranges of {OUTLINE_MIN_PARAGRAPHS} to {OUTLINE_MAX_PARAGRAPHS} paragraphs.
- Plan the complete partition before responding so the final range is also close to the target size.
- The first range must start at paragraph 1.
- Each later range must start immediately after the prior range.
- The final range must end at paragraph {paragraph_count}.
- Include every paragraph exactly once, in ascending order, with no gaps or overlaps.
- Each summary_sentence must be exactly one grammatical sentence synthesizing only its range's paragraph summaries.
- Preserve the Greek names, events, chronology, and elevated tone of the input.
- Do not copy a paragraph summary verbatim when a synthesis is possible.
- Do not return a book summary; the caller will join the summary_sentence values verbatim, in order, to form one prose paragraph.
- JSON strings must escape quotation marks and control characters correctly.

Book: {book_title}
Paragraph summaries as JSON:
{json.dumps(inputs, ensure_ascii=False)}"""


def parse_outline(output: str, paragraph_count: int) -> list[OutlineRange]:
    """Parse and strictly validate a model-generated range outline."""
    try:
        raw: object = json.loads(output)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {error}") from error
    if not isinstance(raw, list) or not raw:
        raise ValueError("outline must be a non-empty JSON array")
    items: list[object] = cast(list[object], raw)
    ranges: list[OutlineRange] = []
    expected_start: int = 1
    expected_keys: set[str] = {"summary_sentence", "start_paragraph", "end_paragraph"}
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict) or set(item) != expected_keys:
            raise ValueError(f"outline item {index} does not have the exact required keys")
        record: dict[str, object] = cast(dict[str, object], item)
        summary: object = record["summary_sentence"]
        start: object = record["start_paragraph"]
        end: object = record["end_paragraph"]
        if not isinstance(summary, str) or not has_requested_shape(summary, "sentence"):
            raise ValueError(f"outline item {index} is not exactly one sentence")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
        ):
            raise ValueError(f"outline item {index} range values must be integers")
        if start != expected_start:
            raise ValueError(f"outline item {index} must start at paragraph {expected_start}")
        size: int = end - start + 1
        if size < 1:
            raise ValueError(f"outline item {index} has an empty or reversed range")
        ranges.append(OutlineRange(summary, start, end))
        expected_start = end + 1
    if expected_start != paragraph_count + 1:
        raise ValueError(f"outline ends at paragraph {expected_start - 1}, not {paragraph_count}")
    return ranges


def outline_cache_key(book_title: str, paragraphs: list[SummarizedParagraph]) -> str:
    """Key an outline by its prompt version and exact paragraph-summary inputs."""
    material: str = json.dumps(
        {
            "source_edition": SOURCE_EDITION,
            "model": MODEL,
            "prompt_version": OUTLINE_PROMPT_VERSION,
            "book_title": book_title,
            "paragraphs": [asdict(paragraph) for paragraph in paragraphs],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def invoke_outline(
    book_title: str, paragraphs: list[SummarizedParagraph], retries: int = 6
) -> tuple[str, list[OutlineRange]]:
    """Invoke Gemma for a structured outline, retrying any invalid response."""
    base_prompt: str = outline_prompt(book_title, paragraphs)
    feedback: str = ""
    last_error: str = "unknown failure"
    for attempt in range(1, retries + 1):
        prompt: str = base_prompt + feedback
        process: subprocess.CompletedProcess[str] = subprocess.run(
            ["bun", "run", "pi", "-p", "--model", MODEL, prompt],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=300,
            check=False,
        )
        output: str = process.stdout.strip()
        if process.returncode == 0:
            try:
                return output, parse_outline(output, len(paragraphs))
            except ValueError as error:
                last_error = str(error)
        else:
            last_error = process.stderr.strip() or f"exit {process.returncode}"
        feedback = (
            f"\n\nYour previous response was invalid because {last_error}. "
            "Rebalance ranges as needed and return the complete corrected JSON array only. "
            f"The invalid response was:\n{output}"
        )
        print(f"Outline attempt {attempt}/{retries} failed for {book_title}: {last_error}", file=sys.stderr)
    raise RuntimeError(f"Outline generation failed for {book_title}: {last_error}")


def outline_cached(
    book_title: str, paragraphs: list[SummarizedParagraph], cache: SummaryCache
) -> list[OutlineRange]:
    """Reuse a validated structured outline or generate and persist one."""
    key: str = outline_cache_key(book_title, paragraphs)
    existing: str | None = cache.get(key)
    if existing is not None:
        try:
            return parse_outline(existing, len(paragraphs))
        except ValueError:
            pass
    output, ranges = invoke_outline(book_title, paragraphs)
    cache.put(key, output)
    return ranges


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
    """Generate cached paragraph summaries and structured book outlines."""
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
        summarized_paragraphs: list[SummarizedParagraph] = [
            SummarizedParagraph(
                number=paragraph.number,
                summary=paragraph_summaries[(book.number, paragraph.number)],
                source=paragraph.source,
            )
            for paragraph in book.paragraphs
        ]
        print(f"Outlining {book.title}...", flush=True)
        ranges: list[OutlineRange] = outline_cached(book.title, summarized_paragraphs, cache)
        sections: list[SummarizedSection] = [
            SummarizedSection(
                number=index,
                summary=outline_range.summary,
                start_paragraph=outline_range.start_paragraph,
                end_paragraph=outline_range.end_paragraph,
                paragraphs=summarized_paragraphs[
                    outline_range.start_paragraph - 1 : outline_range.end_paragraph
                ],
            )
            for index, outline_range in enumerate(ranges, start=1)
        ]
        book_summary: str = " ".join(section.summary for section in sections)
        summarized.append(SummarizedBook(book.number, book.title, book_summary, sections))
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
    if data.get("source_edition") != SOURCE_EDITION or data.get("source_url") != SOURCE_URL:
        raise RuntimeError("Summary data does not match the configured source edition")
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
        raw_sections: object = book.get("sections")
        if not isinstance(raw_sections, list) or not raw_sections:
            raise RuntimeError(f"{source_book.title} does not contain sections")
        sections: list[dict[str, Any]] = cast(list[dict[str, Any]], raw_sections)
        if [section.get("number") for section in sections] != list(range(1, len(sections) + 1)):
            raise RuntimeError(f"Section numbering is invalid in {source_book.title}")
        section_sentences: list[str] = []
        paragraphs: list[dict[str, Any]] = []
        expected_start: int = 1
        for section in sections:
            section_summary: object = section.get("summary")
            start: object = section.get("start_paragraph")
            end: object = section.get("end_paragraph")
            if not isinstance(section_summary, str) or not has_requested_shape(
                section_summary, "sentence"
            ):
                raise RuntimeError(f"Invalid section sentence in {source_book.title}")
            if not isinstance(start, int) or not isinstance(end, int) or start != expected_start:
                raise RuntimeError(f"Invalid section range in {source_book.title}")
            if end < start:
                raise RuntimeError(f"Section range size is invalid in {source_book.title}")
            raw_paragraphs: object = section.get("paragraphs")
            if not isinstance(raw_paragraphs, list) or len(raw_paragraphs) != end - start + 1:
                raise RuntimeError(f"Section paragraph count mismatch in {source_book.title}")
            section_sentences.append(section_summary)
            paragraphs.extend(cast(list[dict[str, Any]], raw_paragraphs))
            expected_start = end + 1
        if book_summary != " ".join(section_sentences):
            raise RuntimeError(f"Book summary is not the verbatim section join in {source_book.title}")
        if len(paragraphs) != len(source_book.paragraphs):
            raise RuntimeError(f"Paragraph count mismatch in {source_book.title}")
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
    expected_count: int = sum(len(book.paragraphs) for book in source_books)
    if summary_count != expected_count:
        raise RuntimeError(f"Expected {expected_count} paragraph nodes, found {summary_count}")
    html: str = OUTPUT_PATH.read_text(encoding="utf-8")
    if "__ODYSSEY_DATA__" in html or html.count("const ODYSSEY_DATA = ") != 1:
        raise RuntimeError("Standalone HTML does not contain exactly one embedded data set")
    if f'const SOURCE_EDITION = "{SOURCE_EDITION}";' not in html or SOURCE_URL not in html:
        raise RuntimeError("Standalone HTML does not contain the configured source edition")
    if "#linknote-${match[1]}" not in html or 'className = "footnote"' not in html:
        raise RuntimeError("Standalone HTML does not contain the Gutenberg footnote renderer")
    if "<script src=" in html or "<link rel=\"stylesheet\"" in html:
        raise RuntimeError("Standalone HTML unexpectedly depends on an external script or stylesheet")
    print(
        f"Verified 24 books, {sum(len(cast(list[object], book['sections'])) for book in books)} "
        f"clickable sections, {expected_count} sentence/source nodes, and standalone HTML."
    )


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
        "source_edition": SOURCE_EDITION,
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
        "hierarchy_version": 2,
        "section_paragraph_range": [OUTLINE_MIN_PARAGRAPHS, OUTLINE_MAX_PARAGRAPHS],
        "source_edition": SOURCE_EDITION,
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
