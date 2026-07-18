This folder is empty. Your task is to create a Python program that produces a hierarchical representation of The Odyssey ("the source work"). Skip the introductions and footnotes, instead only focus on BOOK I through BOOK XXIV. You must also produce a single HTML file that contains a treeview viewer for the results.

- Use only python3 & bash. 
- When using python, you MUST use a venv and use it exclusively it to install dependencies. You MUST use type hints in all Python code. 
- You MUST NOT install any system dependencies.
- Use Project Gutenberg's public domain copy of the work: https://www.gutenberg.org/cache/epub/1728/pg1728-images.html Leave a copy of the original file(s) here in the repo. 
- Create a single HTML file with a minimal tree control using only vanilla JS, barebones CSS, and HTML. We're not trying to design anything beautiful, but rather summarize text. 
- Summarize each book into one paragraph. 
- When clicking on the book paragraph, expand to a summary view where each paragraph from the source is summarized to one sentence.
- When clicking the sentence, show the source paragraph. 
- To repeat: the final work will be 24 paragraphs, one for each Book. Clicking any paragraph expands to the summarized book where each paragraph has been condensed to one sentence. Clicking each sentence yields the original source paragraph.
- To summarize the work, do not do it yourself, use this command locally: `bun run pi -p --model "gemma-4-26B-A4B" --thinking off "Summarize this text into one  {sentence|paragraph}. DO NOT include any preamble, ONLY respond with the summary. Preserve the source style or tone: '{source text}'"` 
- Validate the summarization prompt a few times before running it over the whole source work. 
- The download & summarization work must be repeatable. The Python program(s) must download source material from Gutenberg, parse it, invoke the summarization commands, store the summaries somehow (probably json). 
- This is not true for your HTML / JS foundation, it can live in the repo and read the parsed summarizations. 