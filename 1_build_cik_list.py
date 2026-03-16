# “””
1_build_cik_list.py

Downloads the SEC EDGAR submissions.zip bulk file (one-shot, ~1GB) and
produces two output files:

cik_list.txt        — one CIK per line, for edgar-crawler’s config.json
cik_metadata.jsonl  — CIK, company name, SIC code, tickers, state

Filter by SIC code ranges or specific codes via CLI flags.

Usage:
python 1_build_cik_list.py
python 1_build_cik_list.py –sic-codes 7372 5411 3674
python 1_build_cik_list.py –sic-range 7000 7999
python 1_build_cik_list.py –sic-range 2000 3999 –sic-range 7000 8999
“””

import argparse
import io
import json
import logging
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format=”%(asctime)s %(levelname)s %(message)s”)
log = logging.getLogger(**name**)

# !! Replace with your real name/email — SEC EDGAR requires this !!

USER_AGENT = “YourName yourname@example.com”

SUBMISSIONS_ZIP_URL = (
“https://data.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip”
)

OUTPUT_DIR = Path(“data”)

def download_submissions_zip(cache_path: Path) -> bytes:
“”“Download submissions.zip, using a local cache if already present.”””
if cache_path.exists():
log.info(f”Using cached submissions.zip at {cache_path}”)
return cache_path.read_bytes()

```
log.info("Downloading submissions.zip (~1 GB) — this will take a few minutes...")
r = requests.get(
    SUBMISSIONS_ZIP_URL,
    headers={"User-Agent": USER_AGENT},
    stream=True,
    timeout=120,
)
r.raise_for_status()

total = int(r.headers.get("content-length", 0))
buf = io.BytesIO()
with tqdm(total=total, unit="B", unit_scale=True, desc="submissions.zip") as pbar:
    for chunk in r.iter_content(chunk_size=1 << 20):
        buf.write(chunk)
        pbar.update(len(chunk))

data = buf.getvalue()
cache_path.parent.mkdir(parents=True, exist_ok=True)
cache_path.write_bytes(data)
log.info(f"Saved to {cache_path}")
return data
```

def sic_in_filter(sic: str, sic_codes: set, sic_ranges: list) -> bool:
“”“Return True if the SIC code passes any of the filters.”””
# No filters = accept everything
if not sic_codes and not sic_ranges:
return True
if not sic:
return False
try:
sic_int = int(sic)
except ValueError:
return False
if sic in sic_codes:
return True
for lo, hi in sic_ranges:
if lo <= sic_int <= hi:
return True
return False

def build_cik_list(
raw_zip: bytes,
sic_codes: set,
sic_ranges: list,
output_dir: Path,
) -> None:
output_dir.mkdir(parents=True, exist_ok=True)
cik_list_path = output_dir / “cik_list.txt”
metadata_path = output_dir / “cik_metadata.jsonl”

```
log.info("Scanning submissions.zip...")
accepted = []

with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
    names = zf.namelist()
    log.info(f"  {len(names):,} entries in zip")

    for name in tqdm(names, desc="Filtering"):
        if not name.endswith(".json"):
            continue
        try:
            data = json.loads(zf.read(name))
        except Exception:
            continue

        sic = str(data.get("sic", ""))
        if not sic_in_filter(sic, sic_codes, sic_ranges):
            continue

        cik = str(data.get("cik", "")).lstrip("0")
        if not cik:
            continue

        accepted.append(
            {
                "cik": cik,
                "company_name": data.get("name", ""),
                "sic_code": sic,
                "tickers": data.get("tickers", []),
                "state_of_inc": data.get("stateOfIncorporation", ""),
                "state_location": data.get("stateOfLocation", ""),
            }
        )

log.info(f"  {len(accepted):,} companies matched filters")

with open(cik_list_path, "w") as f:
    for rec in accepted:
        f.write(rec["cik"] + "\n")

with open(metadata_path, "w") as f:
    for rec in accepted:
        f.write(json.dumps(rec) + "\n")

log.info(f"Written: {cik_list_path}")
log.info(f"Written: {metadata_path}")
log.info(
    "\nNext step: update edgar-crawler/config.json with the path to cik_list.txt, "
    "then run: python edgar-crawler/download_filings.py"
)
```

if **name** == “**main**”:
parser = argparse.ArgumentParser(
description=“Build CIK list from EDGAR submissions.zip”
)
parser.add_argument(
“–sic-codes”,
nargs=”+”,
default=[],
help=“Specific SIC codes to include, e.g. –sic-codes 7372 5411”,
)
parser.add_argument(
“–sic-range”,
nargs=2,
type=int,
metavar=(“START”, “END”),
action=“append”,
default=[],
help=“SIC code range (inclusive). Repeatable: –sic-range 2000 3999 –sic-range 7000 8999”,
)
parser.add_argument(
“–no-cache”,
action=“store_true”,
help=“Re-download submissions.zip even if cached”,
)
args = parser.parse_args()

```
cache_path = OUTPUT_DIR / "submissions.zip"
if args.no_cache and cache_path.exists():
    cache_path.unlink()

raw = download_submissions_zip(cache_path)

build_cik_list(
    raw_zip=raw,
    sic_codes=set(args.sic_codes),
    sic_ranges=args.sic_range,
    output_dir=OUTPUT_DIR,
)
```
