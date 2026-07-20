# The Odyssey tree viewer

View it live here: https://pettijohn.github.io/homer-odyssey-tree/

This project downloads Project Gutenberg's public-domain `pg1728` HTML edition
of *The Odyssey*, extracts Books I–XXIV, summarizes each source paragraph into one
sentence, groups those paragraphs into model-generated ranges, and builds a
standalone tree-view document.

The finished viewer is `index.html`. It has no runtime dependencies and can
be opened directly in a browser. Each book begins with one summary paragraph.
Select any sentence in that paragraph to reveal the summaries for its group of
usually 5–10 source paragraphs, then select a paragraph summary to reveal its
source. The 5–10 range is guidance rather than a hard limit, so Gemma may choose
a different boundary to preserve a coherent narrative unit.
The hard-coded `SOURCE_EDITION` setting near the top of its script defaults to
`pg1728`; its matching URL is used for all linked footnote references.

## Rebuild

Python dependencies must stay inside the virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python pipeline.py all
```

The `all` command validates the summarization prompt three times before bulk
generation. Gemma returns a validated JSON range outline for each book; Python
joins its section sentences verbatim to form the book paragraph. Generated model responses are saved immediately in
`data/summary-cache.json`, so interrupted runs resume without repeating work.
Gemma's thinking mode is left enabled for new model calls.
Four model requests run concurrently by default; use `--workers 1` for a
strictly serial run. Other repeatable stages are `download`, `parse`,
`validate`, `summarize`, `build`, and `verify`.

Artifacts:

- `source/pg1728-images.html`: untouched active Gutenberg source download
- `data/parsed.json`: Books I–XXIV before summarization
- `data/summary-cache.json`: resumable content-addressed model cache
- `data/odyssey.json`: complete hierarchical data
- `index.html`: standalone interactive viewer with embedded data
