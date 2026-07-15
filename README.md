# The Odyssey tree viewer

View it live here: https://pettijohn.github.io/homer-odyssey-tree/

This project downloads Project Gutenberg's public-domain HTML edition of *The
Odyssey*, extracts Books I–XXIV, summarizes each source paragraph into one
sentence and each book into one paragraph with the required local model, and
builds a standalone tree-view document.

The finished viewer is `index.html`. It has no runtime dependencies and can
be opened directly in a browser. Select a book paragraph to see its sentence
summaries, then select a sentence to see the original paragraph.

## Rebuild

Python dependencies must stay inside the virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python pipeline.py all
```

The `all` command validates the summarization prompt three times before bulk
generation. Generated model responses are saved immediately in
`data/summary-cache.json`, so interrupted runs resume without repeating work.
Four model requests run concurrently by default; use `--workers 1` for a
strictly serial run. Other repeatable stages are `download`, `parse`,
`validate`, `summarize`, `build`, and `verify`.

Artifacts:

- `source/pg1727-images.html`: untouched Gutenberg source download
- `data/parsed.json`: Books I–XXIV before summarization
- `data/summary-cache.json`: resumable content-addressed model cache
- `data/odyssey.json`: complete hierarchical data
- `index.html`: standalone interactive viewer with embedded data
