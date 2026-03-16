"""
4_export_csv.py

Converts data/dataset.jsonl → data/dataset.csv (or custom paths via CLI).

Usage:
  python 4_export_csv.py
  python 4_export_csv.py --input data/dataset.jsonl --output data/dataset.csv
"""

import argparse
import csv
import json
from pathlib import Path

FIELDNAMES = [
    "cik",
    "company_name",
    "sic_code",
    "nace_code",
    "filing_type",
    "filing_date",
    "period_of_report",
    "state_of_inc",
    "filing_url",
    "description",
]


def jsonl_to_csv(input_path: Path, output_path: Path) -> int:
    """Convert a JSONL dataset file to CSV. Returns the number of rows written."""
    count = 0
    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                writer.writerow(row)
                count += 1
            except Exception:
                pass
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset.jsonl to CSV")
    parser.add_argument("--input",  type=Path, default=Path("data/dataset.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/dataset.csv"))
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}")
        raise SystemExit(1)

    n = jsonl_to_csv(args.input, args.output)
    print(f"Wrote {n:,} rows → {args.output}")
