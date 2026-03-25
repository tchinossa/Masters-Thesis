# Masters-Thesis
Thesis + Code for all analysis performed for my Masters' thesis (Framing and Bias in Media Coverage of the Israel – Palestine Conflict after October 7, 2023: A Comparative NLP-Based Analysis of CNN and Al Jazeera)


This repository contains the end-to-end pipeline used in my Master’s thesis to collect news articles from CNN and Al Jazeera and compare outlet-level traits using:
- Scraping + preprocessing
- Lexicon/VADER sentiment
- Word2Vec neighbor overlap (Jaccard)
- SBERT semantic similarity / neighbor overlap
- LLM-based structured extraction (frames, bias signals, historical context, actors)
- Aggregation into outlet-level comparison tables

### 1) Data collection (scraping)
- **`CNN Scraper.ipynb`**
  Notebook to scrape CNN articles, store raw content/metadata (e.g., title/date/url/text), and export to local files for downstream NLP.

- **`Scraper Al Jazeera.ipynb`**
  Notebook to scrape Al Jazeera articles, store raw content/metadata, and export to local files for downstream NLP.

### 2) NLP baseline (lexicon, VADER, Word2Vec)
- **`NLP Analysis (lexicon, VADER, word2vec).ipynb`**
  Notebook containing classical NLP analysis: preprocessing, lexicon-based measures, VADER sentiment, and Word2Vec training/evaluation, including neighbor overlap outputs.

### 3) SBERT semantic analysis
- **`SBERT.py`**
  Script for SBERT-based processing: embedding articles/sentences, retrieving neighbors, computing overlap and similarity measures (e.g., Jaccard overlap across outlets), and exporting structured JSON outputs.

### 4) LLM-based structured extraction
- **`LLM_Analysis_final.py`**
  Main script to run the LLM extraction over articles and produce per-article JSON outputs with a consistent schema (frames, bias signals/evidence spans, historical context, and quant-ready notes).

  Typical output fields include:
  - `primary_frame`, `secondary_frames`
  - `bias_signals` (typed signals + evidence spans)
  - `historical_context` (level + references)
  - chunk-level `notes_for_quant_analysis` (actors as agents/victims, salient labels)
