"""
LLM_Analysis
Requirements:
    ollama pull mistral
    ollama serve
"""

import os
import re
import json
import time
import logging
import requests
import pandas as pd
from tqdm import tqdm

# -------------------------
# CONFIG
# -------------------------

OLLAMA_MODEL = "mistral"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

OUTPUT_DIR = "./results/llm_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHUNK_CHARS = 6000
CHUNK_OVERLAP = 500

TEMPERATURE = 0.0
MAX_TOKENS_GENERATED = 1200
NUM_CTX = 4096

MAX_RETRIES = 3
REQUEST_SLEEP = 0.05

SMOKE_TEST_N = None   # set to None for full run

logging.basicConfig(level=logging.INFO)


# -------------------------
# PROMPT
# -------------------------

COMBINED_PROMPT_TEMPLATE = r"""
SYSTEM: You are a research annotation assistant for a Master's thesis comparing CNN vs Al Jazeera coverage of the Israel–Palestine conflict after Oct 7, 2023.
Your job is to:
  A) LABEL the dominant FRAME
  B) EXTRACT LANGUAGE-LEVEL BIAS SIGNALS
  C) ASSESS whether the article provides HISTORICAL CONTEXT
You do NOT evaluate truth. You do NOT use external knowledge. You ONLY use the provided text.

OUTPUT RULES (mandatory):
1) Return ONLY ONE valid JSON object. No prose, no markdown, no backticks.
2) Use ONLY the allowed labels exactly as written. Do not invent new labels.
3) All evidence spans MUST be verbatim substrings from ARTICLE_TEXT.
4) Offsets are CHARACTER OFFSETS in ARTICLE_TEXT (0-based, end-exclusive).
   If you are not 100% sure about offsets, set start=-1 and end=-1 (do NOT guess).
5) Keep it short: <=3 frame_evidence spans, <=8 bias_signals, <=5 historical_references.

────────────────────────────────────────
A) FRAMING
────────────────────────────────────────

ALLOWED PRIMARY FRAME (choose ONE):
- Military self-defense
- Terrorism and security
- Humanitarian crisis
- International law and accountability
- Resistance and liberation
- Diplomatic and political process
- Neutral reporting
- Other

FRAME DEFINITIONS:
- Military self-defense: actions framed as defensive response, military operation, retaliation, security necessity by state actors.
- Terrorism and security: events framed as terrorism, security threat, attacks on civilians, hostage-taking, criminalization language.
- Humanitarian crisis: emphasis on civilian suffering, casualties, displacement, aid, hospitals, shortages, hunger, destruction of living conditions.
- International law and accountability: focus on legality, war crimes, investigations, courts, human rights law, UN/ICJ/ICC, accountability.
- Resistance and liberation: actions framed as resistance against occupation, liberation struggle, right to self-determination.
- Diplomatic and political process: focus on negotiations, diplomacy, ceasefires, political maneuvering, international relations.
- Neutral reporting: mostly factual/attributed reporting, balanced sourcing, low evaluative language.
- Other: none of the above dominates.

Also choose up to TWO SECONDARY FRAMES from the same list (or leave empty).

────────────────────────────────────────
B) BIAS SIGNALS
────────────────────────────────────────

Choose signal_type ONLY from:
- loaded_language (emotionally charged adjectives/adverbs; moral judgment terms)
- labeling (militants/fighters/terrorists; "occupation"; "massacre"; "genocide"; etc.)
- agency_assignment (who is agent vs victim; who "does" vs who "is affected")
- passive_or_agentless (passive voice or missing agent: "were killed", "died", "in the strike")
- sourcing_pattern (anonymous sources vs named; heavy reliance on one side; "officials said")
- hedging_or_certainty (may/allegedly/claims vs definitive certainty)
- casualty_framing (whose casualties counted/emphasized; precision vs vagueness)
- emphasis_omission (major context missing or backgrounded; note what is absent)
- metaphor_or_analogy (animalizing/dehumanizing metaphors; historical analogies)
- false_equivalence (presenting asymmetric situations as equivalent)

IMPORTANT:
- A bias signal is a *linguistic feature* that could influence perception. Not a factual claim.
- Each signal must cite a short evidence span (5–40 words) copied verbatim.

────────────────────────────────────────
C) HISTORICAL CONTEXT
────────────────────────────────────────

Assess whether the article provides historical background to contextualize the events reported.
Look for:
- References to past events (e.g., 1948 Nakba, 1967 war, Oslo Accords, previous Gaza wars, settlement history, etc.)
- Mentions of historical patterns, long-term occupation, blockade history
- Any backgrounding that goes beyond the immediate news event

Rate historical_context_level as ONE of:
- none: no historical context at all; article treats events in isolation
- minimal: brief passing mention (e.g., "the decades-long conflict") without specifics
- moderate: some specific historical references that help contextualize
- extensive: detailed historical background integrated into the reporting

────────────────────────────────────────
FEW-SHOT MINI EXAMPLES (format only):
────────────────────────────────────────

Example Frame:
Text: "The army said it launched strikes after rockets were fired."
=> Military self-defense

Example Bias signals:
Span: "brutal attack" => loaded_language
Span: "were killed" => passive_or_agentless
Span: "officials said" => sourcing_pattern
Span: "terrorists" => labeling

Example Historical context:
Span: "since the 1967 occupation of the West Bank" => reference to 1967 war
Level: moderate

────────────────────────────────────────
INPUT:
────────────────────────────────────────

ARTICLE_TITLE: {title}
ARTICLE_TEXT: {content}

────────────────────────────────────────
OUTPUT JSON SCHEMA (exact keys):
────────────────────────────────────────

{{
  "primary_frame": "",
  "secondary_frames": [],
  "frame_confidence": 0.0,
  "frame_evidence": [
    {{"span": "", "start": -1, "end": -1, "why": "focus|quote|lexical"}}
  ],
  "bias_signals": [
    {{"signal_type": "", "span": "", "start": -1, "end": -1, "note": ""}}
  ],
  "historical_context": {{
    "historical_context_level": "none|minimal|moderate|extensive",
    "historical_references": [
      {{"span": "", "start": -1, "end": -1, "event_referenced": "", "period": ""}}
    ]
  }},
  "notes_for_quant_analysis": {{
    "salient_labels_used": [],
    "main_actors_as_agents": [],
    "main_actors_as_victims": []
  }}
}}
"""


# -------------------------
# UTILS
# -------------------------

def chunk_text(text):
    chunks = []
    i = 0
    while i < len(text):
        end = min(len(text), i + CHUNK_CHARS)
        chunks.append((i, text[i:end]))
        if end == len(text):
            break
        i = end - CHUNK_OVERLAP
    return chunks


def extract_json(text):
    if not text:
        return None
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except:
        return None


def call_ollama(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS_GENERATED,
            "num_ctx": NUM_CTX
        },
        "stream": False
    }

    for _ in range(MAX_RETRIES):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=600)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except Exception as e:
            logging.warning(f"Ollama call failed: {e}")
            time.sleep(1)

    return None


# -------------------------
# PROCESS ARTICLE
# -------------------------

def process_article(article_id, title, content, date):
    cache_path = os.path.join(OUTPUT_DIR, f"{article_id}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    chunks = chunk_text(content)

    chunk_outputs = []

    for idx, (char_start, chunk_text_content) in enumerate(chunks):

        prompt = COMBINED_PROMPT_TEMPLATE.replace("{title}", title).replace("{content}", chunk_text_content)
        raw = call_ollama(prompt)
        parsed = extract_json(raw)

        if parsed:
            # convert local offsets to global offsets
            for key in ["frame_evidence", "bias_signals"]:
                for item in parsed.get(key, []):
                    if "start" in item and item["start"] != -1:
                        item["start"] += char_start
                        item["end"] += char_start
            # historical references too
            hist = parsed.get("historical_context", {})
            for item in hist.get("historical_references", []):
                if "start" in item and item["start"] != -1:
                    item["start"] += char_start
                    item["end"] += char_start

        chunk_outputs.append(parsed)

        time.sleep(REQUEST_SLEEP)

    # -------- Aggregation --------
    frame_scores = {}
    for c in chunk_outputs:
        if not c:
            continue
        f = c.get("primary_frame")
        conf = float(c.get("frame_confidence", 0.0) or 0.0)
        if f:
            frame_scores.setdefault(f, []).append(conf)

    final_frame = max(frame_scores.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))[0] if frame_scores else "Other"

    # collect all secondary frames
    all_secondary = []
    for c in chunk_outputs:
        if c and "secondary_frames" in c:
            all_secondary.extend(c["secondary_frames"])

    all_bias = []
    for c in chunk_outputs:
        if c and "bias_signals" in c:
            all_bias.extend(c["bias_signals"])

    # aggregate historical context
    all_hist_refs = []
    hist_levels = []
    level_order = {"none": 0, "minimal": 1, "moderate": 2, "extensive": 3}
    for c in chunk_outputs:
        if not c:
            continue
        hist = c.get("historical_context", {})
        lvl = hist.get("historical_context_level", "none")
        hist_levels.append(lvl)
        all_hist_refs.extend(hist.get("historical_references", []))

    final_hist_level = max(hist_levels, key=lambda x: level_order.get(x, 0)) if hist_levels else "none"

    final_result = {
        "article_id": article_id,
        "title": title,
        "date": date,
        "primary_frame": final_frame,
        "secondary_frames": list(set(all_secondary)),
        "bias_signals": all_bias,
        "historical_context": {
            "historical_context_level": final_hist_level,
            "historical_references": all_hist_refs
        },
        "chunks": chunk_outputs
    }

    with open(cache_path, "w") as f:
        json.dump(final_result, f, indent=2)

    return final_result


# -------------------------
# MAIN
# -------------------------

if __name__ == "__main__":

    df_aj = pd.read_csv("aljazeera_cleaned_for_llm.csv")
    df_cnn = pd.read_csv("cnn_cleaned_for_llm.csv")

    if SMOKE_TEST_N:
        df_aj = df_aj.head(SMOKE_TEST_N)
        df_cnn = df_cnn.head(SMOKE_TEST_N)

    for outlet, df in [("cnn", df_cnn), ("aljazeera", df_aj)]:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            process_article(
                f"{outlet}_{idx}",
                str(row["title"]) if pd.notna(row["title"]) else "Untitled",
                str(row["text"]),
                str(row["date"])
            )

    print("DONE.")
