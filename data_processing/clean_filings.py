"""
Step 2: Extract and clean SEC filings, transcript, and press release.
Saves cleaned text files to cleaned/ folder so downstream scripts skip this step.

Output:
  cleaned/manifest.csv          - metadata for all cleaned documents
  cleaned/<accession>.txt       - cleaned narrative text per filing
  cleaned/transcript.txt        - cleaned transcript text
  cleaned/press_release.txt     - cleaned press release text
"""
import os
import re
import json
import math
import pandas as pd
from datetime import datetime
from pypdf import PdfReader
from config import (
    FILINGS_DIR, TRANSCRIPT_PDF, PRESS_RELEASE_TXT,
    CLEANED_DIR, EARNINGS_DATE_STR,
)


# ============================================================
# TEXT EXTRACTION HELPERS
# ============================================================

def get_filing_date(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(100):
            line = f.readline()
            if not line:
                break
            if "FILED AS OF DATE:" in line:
                date_str = line.split(":")[-1].strip()
                return datetime.strptime(date_str, "%Y%m%d")
    return None


def extract_main_document(filepath, form_type):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    doc_pattern = re.compile(
        r"<DOCUMENT>\s*<TYPE>([^\n<]+).*?<TEXT>(.*?)</TEXT>",
        re.DOTALL | re.IGNORECASE,
    )
    for match in doc_pattern.finditer(content):
        doc_type = match.group(1).strip()
        if doc_type.upper().startswith(form_type.upper()):
            return match.group(2)
    return content


def clean_filing_text(raw_text):
    text = raw_text
    text = re.sub(r"</?XBRL>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</?ix:[^>]*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<\?.*?\?>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text, flags=re.DOTALL)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&#160;", " ")
    text = re.sub(r"&#?\w+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_narrative_sections(full_text):
    text_lower = full_text.lower()

    mda_patterns = [
        r"item\s*7[.\s]+management.s discussion and analysis",
        r"item\s*2[.\s]+management.s discussion and analysis",
        r"management.s discussion and analysis",
        r"management discussion and analysis",
        r"item\s*7\.\s*management",
        r"item\s*2\.\s*management",
    ]
    risk_patterns = [
        r"item\s*1a[.\s]+risk\s*factors",
        r"risk\s*factors",
        r"item\s*1a\.\s*risk",
    ]

    item_header_pattern = re.compile(
        r"\bitem\s*\d+[a-z]?\b[.\s]",
        re.IGNORECASE,
    )

    def find_section(patterns, label):
        candidates = []
        for pattern in patterns:
            for match in re.finditer(pattern, text_lower):
                start = match.start()
                search_from = start + 500
                end_match = item_header_pattern.search(text_lower, search_from)
                end = end_match.start() if end_match else min(start + 50000, len(full_text))
                span_len = end - start
                candidates.append((start, end, span_len, pattern))

        if not candidates:
            return None

        best = max(candidates, key=lambda c: c[2])
        start, end, span_len, pattern = best
        section_text = full_text[start:end].strip()
        word_count = len(section_text.split())
        print(f"    [{label}] {word_count} words (best of {len(candidates)} candidates)")
        return section_text

    mda_text = find_section(mda_patterns, "MD&A")
    risk_text = find_section(risk_patterns, "Risk Factors")

    if mda_text is None and risk_text is None:
        print("    [WARN] *** FALLBACK *** No MD&A or Risk Factors found")
        words = full_text.split()
        start_idx = len(words) // 3
        return {
            "mda": None, "risk_factors": None,
            "fallback_used": True,
            "combined_text": " ".join(words[start_idx:]),
        }

    sections = []
    if mda_text:
        sections.append(mda_text)
    if risk_text:
        sections.append(risk_text)

    return {
        "mda": mda_text, "risk_factors": risk_text,
        "fallback_used": False,
        "combined_text": " ".join(sections),
    }


def extract_transcript_text(pdf_path):
    reader = PdfReader(pdf_path)
    pages_text = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            pages_text.append(t)
    return " ".join(pages_text)


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(CLEANED_DIR, exist_ok=True)
    manifest = []

    # --- SEC Filings ---
    print("=" * 60)
    print("CLEANING SEC FILINGS")
    print("=" * 60)

    for form_type in ["10-K", "10-Q"]:
        form_dir = os.path.join(FILINGS_DIR, form_type)
        if not os.path.isdir(form_dir):
            print(f"  [SKIP] No directory for {form_type}")
            continue

        for accession in sorted(os.listdir(form_dir)):
            fpath = os.path.join(form_dir, accession, "full-submission.txt")
            if not os.path.isfile(fpath):
                continue

            date = get_filing_date(fpath)
            if date is None:
                print(f"  [SKIP] No date: {accession}")
                continue

            print(f"\n  {form_type} {date.date()} ({accession})...")
            raw_doc = extract_main_document(fpath, form_type)
            full_clean = clean_filing_text(raw_doc)
            sections = extract_narrative_sections(full_clean)
            narrative = sections["combined_text"]

            mda_words = len(sections["mda"].split()) if sections["mda"] else 0
            risk_words = len(sections["risk_factors"].split()) if sections["risk_factors"] else 0

            # Save cleaned text
            safe_name = accession.replace("/", "_").replace("\\", "_")
            out_path = os.path.join(CLEANED_DIR, f"{safe_name}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(narrative)

            sections_found = []
            if sections["mda"]:
                sections_found.append("MD&A")
            if sections["risk_factors"]:
                sections_found.append("Risk Factors")
            if sections["fallback_used"]:
                sections_found.append("FALLBACK")

            manifest.append({
                "Date": date.strftime("%Y-%m-%d"),
                "Type": form_type,
                "Accession": accession,
                "Filename": f"{safe_name}.txt",
                "WordCount": len(narrative.split()),
                "MDA_Words": mda_words,
                "RiskFactors_Words": risk_words,
                "Sections_Found": "+".join(sections_found) if sections_found else "None",
                "Fallback_Used": sections["fallback_used"],
            })
            print(f"    -> {len(narrative.split())} words saved to {safe_name}.txt")

    # --- Transcript ---
    print(f"\n  Extracting transcript PDF...")
    transcript_text = extract_transcript_text(TRANSCRIPT_PDF)
    out_path = os.path.join(CLEANED_DIR, "transcript.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(transcript_text)

    manifest.append({
        "Date": EARNINGS_DATE_STR,
        "Type": "Transcript",
        "Accession": "JACK-Q4-FY25",
        "Filename": "transcript.txt",
        "WordCount": len(transcript_text.split()),
        "MDA_Words": 0,
        "RiskFactors_Words": 0,
        "Sections_Found": "Full",
        "Fallback_Used": False,
    })
    print(f"    -> {len(transcript_text.split())} words saved to transcript.txt")

    # --- Press Release ---
    print(f"\n  Copying press release...")
    with open(PRESS_RELEASE_TXT, "r", encoding="utf-8") as f:
        pr_text = f.read()
    out_path = os.path.join(CLEANED_DIR, "press_release.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(pr_text)

    manifest.append({
        "Date": EARNINGS_DATE_STR,
        "Type": "PressRelease",
        "Accession": "PR-2025-11-19",
        "Filename": "press_release.txt",
        "WordCount": len(pr_text.split()),
        "MDA_Words": 0,
        "RiskFactors_Words": 0,
        "Sections_Found": "Full",
        "Fallback_Used": False,
    })
    print(f"    -> {len(pr_text.split())} words saved to press_release.txt")

    # --- Save manifest ---
    manifest_df = pd.DataFrame(manifest)
    manifest_path = os.path.join(CLEANED_DIR, "manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\n  Saved manifest: {manifest_path}")
    print(f"  Total documents cleaned: {len(manifest)}")

    print("\n" + "=" * 60)
    print("CLEANING DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
