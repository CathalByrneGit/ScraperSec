# NACE Training Dataset Builder

Builds a labelled dataset of `{company description → NACE code}` pairs from
SEC EDGAR 10-K filings. Designed for training a NACE classification model.

## How it works

```
submissions.zip (bulk, ~1 GB)
        │
        ▼
1_build_cik_list.py          — filter by SIC, output cik_list.txt + metadata
        │
        ▼
edgar-crawler                — download 10-K filings & extract Item 1 text
(download_filings.py
 extract_items.py)
        │
        ▼
3_build_dataset.py           — GLiNER2 extracts description, merge SIC→NACE
        │
        ▼
data/dataset.jsonl           — final training data
```

**Why 10-K and not 10-Q?**
10-K annual reports include a mandatory *Item 1 — Business* section where
companies are required to describe what they do. 10-Q quarterly reports
frequently just say “refer to our Annual Report.” Item 1 is consistently
the right place to find a company description.

**Why `submissions.zip`?**
SEC EDGAR publishes a single zip containing every company’s filing history
and metadata. One download gives you CIK, company name, and SIC code for
every registered filer — no pagination or per-company API calls needed.

-----

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Clone edgar-crawler

```bash
git clone https://github.com/lefterisloukas/edgar-crawler.git
pip install -r edgar-crawler/requirements.txt
```

### 3. Set your User-Agent

Edit `USER_AGENT` at the top of `1_build_cik_list.py`:

```python
USER_AGENT = "YourName yourname@example.com"
```

Also update `user_agent` in `edgar_crawler_config.json`. SEC EDGAR requires
this by policy — use your real name and email.

-----

## Running the pipeline

### Step 1 — Build the CIK list

Downloads `submissions.zip` once (~1 GB, cached locally) and filters to the
SIC codes you want.

```bash
# Specific SIC codes
python 1_build_cik_list.py --sic-codes 7372 5411 3674

# A range of SIC codes (e.g. all manufacturing)
python 1_build_cik_list.py --sic-range 2000 3999

# Multiple ranges (manufacturing + services)
python 1_build_cik_list.py --sic-range 2000 3999 --sic-range 7000 8999

# All companies (no filter — very large)
python 1_build_cik_list.py
```

Outputs:

- `data/cik_list.txt` — one CIK per line
- `data/cik_metadata.jsonl` — CIK, company name, SIC, tickers, state

### Step 2 — Download and extract 10-K filings

Copy the config template and point edgar-crawler at the CIK list:

```bash
cp edgar_crawler_config.json edgar-crawler/config.json
```

Open `edgar-crawler/config.json` and verify `user_agent` is set, then run:

```bash
# Download raw 10-K filings
python edgar-crawler/download_filings.py

# Extract Item 1 from each filing into structured JSON
python edgar-crawler/extract_items.py
```

Extracted JSONs land in `data/EXTRACTED_FILINGS/`. Each file has an `item_1`
field containing clean plain text of the Business section.

### Step 3 — Build the dataset

```bash
# Without SIC→NACE mapping (nace_code field will be null)
python 3_build_dataset.py

# With SIC→NACE mapping
python 3_build_dataset.py --sic-nace-map sic_to_nace.json
```

The script resumes automatically if interrupted — already-processed CIKs
are skipped on re-run.

Output: `data/dataset.jsonl`

-----

## Output format

One JSON record per line:

```json
{
  "cik": "320193",
  "company_name": "Apple Inc.",
  "sic_code": "3571",
  "nace_code": "C26.20",
  "filing_type": "10-K",
  "filing_date": "2023-11-03",
  "period_of_report": "2023-09-30",
  "state_of_inc": "CA",
  "filing_url": "https://www.sec.gov/Archives/edgar/data/...",
  "description": "Apple designs, manufactures and markets smartphones, personal computers, tablets, wearables and accessories, and sells a variety of related services. The Company's products include iPhone, Mac, iPad, AirPods, Apple TV, Apple Watch, Beats products and HomePod. Apple sells its products and services through its retail and online stores and direct sales force, as well as through third-party cellular network carriers, wholesalers, retailers, and resellers."
}
```

-----

## SIC → NACE mapping

Create `sic_to_nace.json` as a flat key-value map:

```json
{
  "3571": "C26.20",
  "7372": "J62.01",
  "5411": "G47.11"
}
```

Eurostat publishes official SIC/NACE concordance tables which are the best
starting point. The two systems don’t map 1:1 — some SIC codes will map to
the same NACE code, and some SIC codes have no clean equivalent. Treat the
mapping as a best-effort label rather than ground truth.

-----

## Tips for scale

- `submissions.zip` is cached after the first download. Re-runs of step 1
  are fast.
- edgar-crawler skips already-downloaded filings by default
  (`skip_present_indices: true`). Safe to interrupt and resume.
- Step 3 also resumes — it skips CIKs already in the output file.
- For the full universe of SEC filers (~15,000 active 10-K filers), expect
  the download step to take several hours. Narrow your SIC filter if you
  want a faster first pass.
- GLiNER2 runs on CPU. Step 3 takes roughly 1–3 seconds per filing.
  10,000 filings ≈ 3–8 hours on a laptop CPU. Consider running overnight.
