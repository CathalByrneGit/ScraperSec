"""
3_build_dataset.py

Reads 10-K JSON files produced by edgar-crawler's extract_items.py,
extracts the company description from Item 1 using GLiNER2, merges in
SIC codes from cik_metadata.jsonl, optionally maps SIC → NACE, and
writes the final training dataset as JSONL.

Usage:
python 3_build_dataset.py
python 3_build_dataset.py --extracted-dir data/EXTRACTED_FILINGS
python 3_build_dataset.py --sic-nace-map sic_to_nace.json --output dataset.jsonl
python 3_build_dataset.py --batch-size 64
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from gliner2 import GLiNER2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULT_EXTRACTED_DIR = Path("data/EXTRACTED_FILINGS")
DEFAULT_METADATA_PATH = Path("data/cik_metadata.jsonl")
DEFAULT_OUTPUT_PATH   = Path("data/dataset.jsonl")
DEFAULT_BATCH_SIZE    = 8   # matches GLiNER2's internal default; increase to 16-32 if memory allows

# GLiNER2 will search this many characters of Item 1.
# Item 1 is already pre-extracted so we don't need 8k of boilerplate —
# 6k covers even the longest business descriptions.
EXTRACTION_CHARS = 6_000

SCHEMA = {
    "company": [
        (
            "description::str::A factual 2-5 sentence passage describing what the "
            "company does, the products or services it provides, and its primary "
            "markets or customers. Exclude financial figures, legal boilerplate, "
            "and forward-looking statements."
        )
    ]
}

# ── Lazy model loading ─────────────────────────────────────────────────────────
# Model is loaded on first call to _get_extractor() rather than at import time,
# so the module can be imported cheaply (e.g. in tests without a GPU/large model).

_EXTRACTOR = None


def _get_extractor() -> GLiNER2:
    """Return the GLiNER2 model, loading it on first use."""
    global _EXTRACTOR
    if _EXTRACTOR is None:
        log.info("Loading GLiNER2 model (fastino/gliner2-base-v1)...")
        _EXTRACTOR = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
        log.info("Model ready.")
    return _EXTRACTOR


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_metadata(path: Path) -> dict:
    """Load cik_metadata.jsonl → {cik: {sic_code, company_name, …}}"""
    meta = {}
    if not path.exists():
        log.warning(f"Metadata file not found: {path} — SIC codes will be missing")
        return meta
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                meta[rec["cik"]] = rec
            except Exception:
                pass
    log.info(f"Loaded metadata for {len(meta):,} companies")
    return meta


def load_nace_map(path: Optional[Path]) -> dict:
    """Load sic_to_nace.json → {sic: nace}. Returns empty dict if not provided."""
    if not path or not path.exists():
        return {}
    with open(path) as f:
        mapping = json.load(f)
    log.info(f"Loaded {len(mapping):,} SIC→NACE mappings")
    return mapping


def extract_description(item1_text: str) -> Optional[str]:
    """Run GLiNER2 on Item 1 text and return extracted description or None."""
    if not item1_text or not item1_text.strip():
        return None
    try:
        result = _get_extractor().extract_json(item1_text[:EXTRACTION_CHARS], SCHEMA)
        records = result.get("company", [])
        if not records:
            return None
        desc = records[0].get("description", "").strip()
        return desc or None
    except Exception as e:
        log.debug(f"GLiNER2 error: {e}")
        return None


def get_item1_text(filing: dict) -> Optional[str]:
    """
    Pull Item 1 text from an edgar-crawler JSON.
    Tries the 10-K key first ('item_1'), falls back to 10-Q part_1_item_1,
    and finally the full part_1 if item-level extraction wasn't possible.
    """
    for key in ("item_1", "part_1_item_1", "part_1"):
        text = filing.get(key, "").strip()
        if text and len(text) > 100:   # skip empty / placeholder values
            return text
    return None


# ── Batch helpers ──────────────────────────────────────────────────────────────

def process_batch(batch: list, nace_map: dict, metadata: dict) -> list:
    """
    Extract descriptions for a batch of (cik, filing, item1_text) tuples using
    GLiNER2's vectorised batch_extract_json(), which runs a single encoder
    forward pass over all texts rather than one pass per filing.

    batch_extract_json() is the underlying method that all single-text helpers
    wrap (extract_json calls batch_extract([text], ..., batch_size=1)[0]).
    Calling it directly for N texts saves N-1 encoder forward passes.
    """
    if not batch:
        return []

    texts = [item1[:EXTRACTION_CHARS] for _, _, item1 in batch]
    raw_results = _get_extractor().batch_extract_json(texts, SCHEMA)

    records = []
    for (cik, filing, _), result in zip(batch, raw_results):
        company_entries = result.get("company", [])
        if not company_entries:
            continue
        description = company_entries[0].get("description", "").strip()
        if not description:
            continue
        meta = metadata.get(cik, {})
        sic_code = filing.get("sic") or meta.get("sic_code", "")
        nace_code = nace_map.get(str(sic_code)) if sic_code else None
        records.append({
            "cik": cik,
            "company_name": filing.get("company") or meta.get("company_name", ""),
            "sic_code": sic_code,
            "nace_code": nace_code,
            "filing_type": filing.get("filing_type", "10-K"),
            "filing_date": filing.get("filing_date", ""),
            "period_of_report": filing.get("period_of_report", ""),
            "state_of_inc": filing.get("state_of_inc") or meta.get("state_of_inc", ""),
            "filing_url": filing.get("htm_filing_link", ""),
            "description": description,
        })
    return records


# ── Main ───────────────────────────────────────────────────────────────────────

def build_dataset(
    extracted_dir: Path,
    metadata_path: Path,
    nace_map_path: Optional[Path],
    output_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    metadata = load_metadata(metadata_path)
    nace_map = load_nace_map(nace_map_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support — skip CIKs already in output
    seen_ciks: set = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    seen_ciks.add(json.loads(line)["cik"])
                except Exception:
                    pass
        log.info(f"Resuming — {len(seen_ciks):,} CIKs already in output")

    json_files = sorted(extracted_dir.glob("**/*.json"))
    log.info(f"Found {len(json_files):,} extracted filing JSONs in {extracted_dir}")

    stats = {"processed": 0, "extracted": 0, "skipped_no_item1": 0, "skipped_no_desc": 0}
    pending_batch: list = []

    def flush_batch(fout) -> None:
        nonlocal pending_batch
        if not pending_batch:
            return
        records = process_batch(pending_batch, nace_map, metadata)
        for rec in records:
            fout.write(json.dumps(rec) + "\n")
        stats["extracted"] += len(records)
        stats["skipped_no_desc"] += len(pending_batch) - len(records)
        pending_batch = []

    with open(output_path, "a") as fout:
        for json_path in tqdm(json_files, desc="Building dataset"):
            try:
                filing = json.loads(json_path.read_text())
            except Exception as e:
                log.debug(f"Failed to read {json_path}: {e}")
                continue

            cik = str(filing.get("cik", "")).lstrip("0")
            if not cik or cik in seen_ciks:
                continue
            seen_ciks.add(cik)
            stats["processed"] += 1

            item1 = get_item1_text(filing)
            if not item1:
                stats["skipped_no_item1"] += 1
                continue

            pending_batch.append((cik, filing, item1))
            if len(pending_batch) >= batch_size:
                flush_batch(fout)
                fout.flush()

        flush_batch(fout)  # flush remainder
        fout.flush()

    log.info("\n── Summary ──────────────────────────────")
    log.info(f"  Filings processed:        {stats['processed']:>6,}")
    log.info(f"  Descriptions extracted:   {stats['extracted']:>6,}")
    log.info(f"  Skipped (no Item 1):      {stats['skipped_no_item1']:>6,}")
    log.info(f"  Skipped (no description): {stats['skipped_no_desc']:>6,}")
    log.info(f"  Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build NACE training dataset from edgar-crawler extracted filings"
    )
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=DEFAULT_EXTRACTED_DIR,
        help="Folder of JSONs from edgar-crawler extract_items.py",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="cik_metadata.jsonl from step 1",
    )
    parser.add_argument(
        "--sic-nace-map",
        type=Path,
        default=None,
        help="JSON file mapping SIC codes to NACE codes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Number of filings to queue before flushing to disk (default: 1). "
            "Increase to reduce I/O overhead for large runs."
        ),
    )
    args = parser.parse_args()

    build_dataset(
        extracted_dir=args.extracted_dir,
        metadata_path=args.metadata,
        nace_map_path=args.sic_nace_map,
        output_path=args.output,
        batch_size=args.batch_size,
    )
