"""
smoke_test.py

End-to-end smoke test for the NACE pipeline.

Creates a set of realistic synthetic SEC 10-K filings (in the same JSON
format edgar-crawler produces), then runs build_dataset() with a simple
sentence-extraction fallback in place of the GLiNER2 model.

Outputs:
  data/smoke/EXTRACTED_FILINGS/*.json  — synthetic filings
  data/smoke/cik_metadata.jsonl        — matching metadata
  data/sample_dataset.jsonl            — raw JSONL output
  data/sample_dataset.csv              — CSV for review / GitHub

Usage:
  python smoke_test.py
"""

import csv
import json
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Synthetic SEC 10-K filings — realistic Item 1 "Business" sections
# ---------------------------------------------------------------------------

FILINGS = [
    {
        "cik": "320193",
        "company": "Apple Inc.",
        "sic": "3571",
        "state_of_inc": "CA",
        "filing_type": "10-K",
        "filing_date": "2023-11-03",
        "period_of_report": "2023-09-30",
        "htm_filing_link": "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm",
        "item_1": (
            "Apple Inc. designs, manufactures, and markets smartphones, personal computers, "
            "tablets, wearables, and accessories, and sells a variety of related services. "
            "The Company's products include iPhone, Mac, iPad, AirPods, Apple TV, Apple Watch, "
            "Beats products, and HomePod. Apple sells its products and services through its "
            "retail and online stores, direct sales force, and third-party cellular network "
            "carriers, wholesalers, retailers, and resellers. The Company's fiscal year ends "
            "on the last Saturday of September. iPhone is the Company's line of smartphones "
            "based on its iOS operating system. Mac is the Company's line of personal computers "
            "based on its macOS operating system. Services include advertising, AppleCare, "
            "cloud services, digital content, and payment services."
        ),
    },
    {
        "cik": "789019",
        "company": "Microsoft Corporation",
        "sic": "7372",
        "state_of_inc": "WA",
        "filing_type": "10-K",
        "filing_date": "2023-07-27",
        "period_of_report": "2023-06-30",
        "htm_filing_link": "https://www.sec.gov/Archives/edgar/data/789019/000078901923000012/msft-20230630.htm",
        "item_1": (
            "Microsoft Corporation develops and supports software, services, devices, and "
            "solutions worldwide. The Company's products include operating systems, cross-device "
            "productivity applications, server applications, business solution applications, "
            "desktop and server management tools, software development tools, and video games. "
            "Microsoft operates through three segments: Productivity and Business Processes, "
            "Intelligent Cloud, and More Personal Computing. The Intelligent Cloud segment "
            "includes public, private, and hybrid server products and cloud services, including "
            "Azure, SQL Server, Windows Server, and GitHub. Microsoft 365 commercial products "
            "and cloud services are sold to businesses through direct sales force and resellers. "
            "LinkedIn connects professionals and enables hiring, learning, and marketing solutions."
        ),
    },
    {
        "cik": "1018724",
        "company": "Amazon.com Inc.",
        "sic": "5961",
        "state_of_inc": "DE",
        "filing_type": "10-K",
        "filing_date": "2024-02-01",
        "period_of_report": "2023-12-31",
        "htm_filing_link": "https://www.sec.gov/Archives/edgar/data/1018724/000101872424000003/amzn-20231231.htm",
        "item_1": (
            "Amazon.com, Inc. engages in the retail sale of consumer products, advertising, "
            "and subscriptions service through online and physical stores in North America "
            "and internationally. The company operates through three segments: North America, "
            "International, and Amazon Web Services (AWS). AWS provides on-demand cloud "
            "computing platforms, storage, and database services to businesses, governments, "
            "and academic institutions. The Company's products and services include merchandise "
            "and content purchased for resale, third-party seller services, subscription services "
            "including Amazon Prime, physical stores such as Whole Foods Market, and advertising "
            "services. Amazon also manufactures and sells electronic devices such as Kindle "
            "e-readers, Fire tablets, Fire TV streaming devices, and Echo smart speakers."
        ),
    },
    {
        "cik": "1326380",
        "company": "Alphabet Inc.",
        "sic": "7372",
        "state_of_inc": "DE",
        "filing_type": "10-K",
        "filing_date": "2024-01-30",
        "period_of_report": "2023-12-31",
        "htm_filing_link": "https://www.sec.gov/Archives/edgar/data/1326380/000032019324000005/goog-20231231.htm",
        "item_1": (
            "Alphabet Inc. is a holding company whose subsidiaries provide a broad range of "
            "products and services. Google Services segment includes Google Search & other, "
            "YouTube ads, Google Network Members' properties, Google subscriptions, platforms "
            "and devices. Google Cloud segment includes Google Cloud Platform and Google "
            "Workspace. Alphabet's Other Bets segment includes businesses such as Waymo "
            "self-driving technology, Verily life sciences, and Wing drone delivery. "
            "Google's advertising revenue is generated primarily from the display of "
            "performance and brand advertising. The Company provides consumers with products "
            "and services across devices including smartphones, tablets, and personal computers."
        ),
    },
    {
        "cik": "1318605",
        "company": "Tesla Inc.",
        "sic": "3711",
        "state_of_inc": "DE",
        "filing_type": "10-K",
        "filing_date": "2024-01-26",
        "period_of_report": "2023-12-31",
        "htm_filing_link": "https://www.sec.gov/Archives/edgar/data/1318605/000095017024002681/tsla-20231231.htm",
        "item_1": (
            "Tesla, Inc. designs, develops, manufactures, leases, and sells electric vehicles, "
            "energy generation and storage systems, and related services. The Company operates "
            "through two segments: Automotive, and Energy Generation and Storage. The Automotive "
            "segment includes the design, development, manufacturing, sales, and leasing of "
            "electric vehicles, including Model 3, Model Y, Model S, Model X, Cybertruck, "
            "and Tesla Semi. The Energy Generation and Storage segment includes the design, "
            "manufacture, installation, sales, and leasing of solar energy generation and "
            "energy storage products and related services to customers, including residential, "
            "commercial, and industrial customers, and utilities. The Company's Supercharger "
            "network provides fast charging for Tesla vehicles."
        ),
    },
    {
        "cik": "200406",
        "company": "Johnson & Johnson",
        "sic": "2836",
        "state_of_inc": "NJ",
        "filing_type": "10-K",
        "filing_date": "2024-02-14",
        "period_of_report": "2023-12-31",
        "htm_filing_link": "https://www.sec.gov/Archives/edgar/data/200406/000020040624000002/jnj-20231231.htm",
        "item_1": (
            "Johnson & Johnson is a holding company that, together with its subsidiaries, "
            "researches, develops, manufactures, and sells a broad range of products in the "
            "health care field. The Company operates through two segments: Innovative Medicine "
            "and MedTech. The Innovative Medicine segment includes pharmaceutical products "
            "in the areas of immunology, infectious diseases, neuroscience, oncology, "
            "cardiovascular and metabolism, and pulmonary hypertension. The MedTech segment "
            "includes electrophysiology, heart recovery, vision, interventional solutions, "
            "orthopaedics, and surgery products. Products are sold to wholesalers, hospitals, "
            "retailers, physicians, nurses, therapists, and consumers worldwide."
        ),
    },
    {
        "cik": "34088",
        "company": "Exxon Mobil Corporation",
        "sic": "2911",
        "state_of_inc": "NJ",
        "filing_type": "10-K",
        "filing_date": "2024-02-23",
        "period_of_report": "2023-12-31",
        "htm_filing_link": "https://www.sec.gov/Archives/edgar/data/34088/000003408824000006/xom-20231231.htm",
        "item_1": (
            "Exxon Mobil Corporation explores for and produces crude oil and natural gas in "
            "the United States and other countries. The Company manufactures petroleum products "
            "and transports and sells crude oil, natural gas, and petroleum products. Exxon "
            "Mobil operates through four segments: Upstream, Energy Products, Chemical "
            "Products, and Specialty Products. The Upstream segment explores for and produces "
            "crude oil, natural gas, and bitumen through wholly-owned subsidiaries and "
            "equity companies. The Company's refining and supply operations produce petroleum "
            "products including gasoline, diesel, and jet fuel. The Chemical Products segment "
            "manufactures and markets petrochemicals including olefins, polyolefins, "
            "aromatics, and a variety of other specialty products globally."
        ),
    },
    {
        "cik": "731642",
        "company": "Walmart Inc.",
        "sic": "5331",
        "state_of_inc": "DE",
        "filing_type": "10-K",
        "filing_date": "2024-03-19",
        "period_of_report": "2024-01-31",
        "htm_filing_link": "https://www.sec.gov/Archives/edgar/data/731642/000073164224000012/wmt-20240131.htm",
        "item_1": (
            "Walmart Inc. engages in the retail and wholesale operations in various formats "
            "worldwide. The Company operates through three segments: Walmart U.S., Walmart "
            "International, and Sam's Club. Walmart U.S. segment operates as a retailer in "
            "the United States, including supercenters, discount stores, neighborhood markets, "
            "and e-commerce. The International segment consists of operations in 18 countries "
            "and includes numerous formats such as supercenters, warehouse clubs, and cash "
            "and carry operations. Sam's Club segment operates membership-only warehouse "
            "clubs and e-commerce in the United States. The Company offers merchandise "
            "including groceries, health and wellness, technology, entertainment, and apparel."
        ),
    },
    {
        "cik": "1403161",
        "company": "Meta Platforms Inc.",
        "sic": "7370",
        "state_of_inc": "DE",
        "filing_type": "10-K",
        "filing_date": "2024-02-02",
        "period_of_report": "2023-12-31",
        "htm_filing_link": "https://www.sec.gov/Archives/edgar/data/1403161/000140316124000005/meta-20231231.htm",
        "item_1": (
            "Meta Platforms, Inc. develops products that enable people to connect and share "
            "with friends and family through mobile devices, personal computers, virtual reality "
            "headsets, and wearables. The Company operates through two segments: Family of Apps "
            "and Reality Labs. Family of Apps includes Facebook, Instagram, Messenger, WhatsApp, "
            "and other services. Reality Labs includes augmented and virtual reality related "
            "consumer hardware, software, and content. The Company generates revenue primarily "
            "from selling advertising placements to marketers. Ads are displayed across the "
            "Company's products including Facebook, Instagram, Messenger, and third-party "
            "applications that are part of the Meta Audience Network."
        ),
    },
    {
        "cik": "12927",
        "company": "Boeing Company",
        "sic": "3728",
        "state_of_inc": "DE",
        "filing_type": "10-K",
        "filing_date": "2024-01-31",
        "period_of_report": "2023-12-31",
        "htm_filing_link": "https://www.sec.gov/Archives/edgar/data/12927/000001292724000005/ba-20231231.htm",
        "item_1": (
            "The Boeing Company is an aerospace and defense manufacturer that designs, "
            "develops, manufactures, sells, services, and supports commercial jetliners, "
            "military aircraft, satellites, missile defense, human space flight, and launch "
            "systems and services. The Company operates through four segments: Commercial "
            "Airplanes, Defense, Space & Security, Global Services, and Boeing Capital. "
            "Commercial Airplanes develops, produces, and markets commercial jet aircraft "
            "principally to the world's airlines, leasing companies, and governments. "
            "Defense, Space & Security engages in the research, development, production, "
            "and modification of manned and unmanned military aircraft and weapons systems. "
            "Global Services provides services to commercial and government customers worldwide."
        ),
    },
    {
        "cik": "1067983",
        "company": "Berkshire Hathaway Inc.",
        "sic": "6311",
        "state_of_inc": "DE",
        "filing_type": "10-K",
        "filing_date": "2024-02-24",
        "period_of_report": "2023-12-31",
        "htm_filing_link": "https://www.sec.gov/Archives/edgar/data/1067983/000010679824000003/brka-20231231.htm",
        "item_1": (
            "Berkshire Hathaway Inc. is a holding company owning subsidiaries engaged in "
            "a number of diverse business activities. The Company's operating businesses "
            "include insurance and reinsurance, freight rail transportation, utilities and "
            "energy, manufacturing, service and retailing. Insurance and reinsurance operations "
            "are conducted through GEICO, Berkshire Hathaway Reinsurance Group, and General Re. "
            "The BNSF Railway is one of the largest freight rail networks in North America. "
            "Berkshire Hathaway Energy operates regulated electric and gas utilities. "
            "Manufacturing operations include industrial, building, and consumer products. "
            "The Company also makes significant investments in publicly traded equity securities."
        ),
    },
    {
        "cik": "1090012",
        "company": "NVIDIA Corporation",
        "sic": "3674",
        "state_of_inc": "DE",
        "filing_type": "10-K",
        "filing_date": "2024-02-21",
        "period_of_report": "2024-01-28",
        "htm_filing_link": "https://www.sec.gov/Archives/edgar/data/1090012/000109001224000004/nvda-20240128.htm",
        "item_1": (
            "NVIDIA Corporation provides graphics, compute, and networking solutions in "
            "the United States, Taiwan, China, and internationally. The Company operates "
            "through two segments: Compute & Networking and Graphics. The Compute & "
            "Networking segment includes data center accelerated computing platforms and "
            "end-to-end networking platforms, automotive platforms, and Jetson for robotics "
            "and embedded edge computing. The Graphics segment includes GeForce GPUs for "
            "gaming and PCs, Quadro and NVIDIA RTX GPUs for enterprise design, and SHIELD "
            "for gaming and streaming. NVIDIA's data center products are used to accelerate "
            "artificial intelligence workloads including large language models and generative AI. "
            "The Company sells its products to original equipment manufacturers, original design "
            "manufacturers, system builders, cloud service providers, and automotive manufacturers."
        ),
    },
    {
        "cik": "51143",
        "company": "International Business Machines Corporation",
        "sic": "7372",
        "state_of_inc": "NY",
        "filing_type": "10-K",
        "filing_date": "2024-01-24",
        "period_of_report": "2023-12-31",
        "htm_filing_link": "https://www.sec.gov/Archives/edgar/data/51143/000005114324000002/ibm-20231231.htm",
        "item_1": (
            "International Business Machines Corporation provides integrated solutions and "
            "services worldwide. The Company operates through two segments: Software and "
            "Consulting. The Software segment includes hybrid platform and solutions such as "
            "Red Hat OpenShift, transaction processing software, and automation and data "
            "management software. The Consulting segment provides business transformation "
            "services, technology consulting, and application operations. IBM's products and "
            "services span cloud and data platforms, AI and automation, security, and "
            "infrastructure. The Company serves clients in financial services, healthcare, "
            "government, telecommunications, and manufacturing industries globally."
        ),
    },
    {
        "cik": "877890",
        "company": "Pfizer Inc.",
        "sic": "2836",
        "state_of_inc": "DE",
        "filing_type": "10-K",
        "filing_date": "2024-01-30",
        "period_of_report": "2023-12-31",
        "htm_filing_link": "https://www.sec.gov/Archives/edgar/data/877890/000087789024000003/pfe-20231231.htm",
        "item_1": (
            "Pfizer Inc. discovers, develops, manufactures, markets, distributes, and sells "
            "biopharmaceutical products worldwide. The Company operates through two segments: "
            "Biopharma and Pfizer CentreOne. Biopharma includes primary care, oncology, "
            "hospital, inflammation and immunology, and vaccines products. The Company's "
            "marketed products include vaccines for COVID-19 and pneumococcal disease, "
            "treatments for cancer, cardiovascular disease, rare diseases, and hospital-based "
            "infections. Pfizer's research and development focuses on novel targets and "
            "modalities including small molecules, biologics, and gene therapies. Products "
            "are sold to wholesalers, retailers, hospitals, clinics, and government agencies."
        ),
    },
    {
        "cik": "40987",
        "company": "General Electric Company",
        "sic": "3612",
        "state_of_inc": "NY",
        "filing_type": "10-K",
        "filing_date": "2024-02-23",
        "period_of_report": "2023-12-31",
        "htm_filing_link": "https://www.sec.gov/Archives/edgar/data/40987/000004098724000004/ge-20231231.htm",
        "item_1": (
            "General Electric Company operates as a high-tech industrial company focused "
            "on aviation and energy transition. The Company operates through two segments: "
            "GE Aerospace and GE Vernova. GE Aerospace designs, manufactures, and services "
            "commercial and military aircraft engines, integrated engine components, and "
            "electric power systems. GE Vernova includes GE Power, which provides equipment "
            "and services for gas, steam, and hydroelectric power generation, and GE "
            "Renewable Energy, which provides wind and solar energy solutions. The Company "
            "serves commercial airlines, military customers, electric utilities, and "
            "independent power producers globally."
        ),
    },
    {
        "cik": "1001012",
        "company": "Chevron Corporation",
        "sic": "2911",
        "state_of_inc": "DE",
        "filing_type": "10-K",
        "filing_date": "2024-02-23",
        "period_of_report": "2023-12-31",
        "htm_filing_link": "https://www.sec.gov/Archives/edgar/data/1001012/000100101224000003/cvx-20231231.htm",
        "item_1": (
            "Chevron Corporation is an integrated energy company engaged in every aspect "
            "of the oil, natural gas, and geothermal energy industries. The Company explores "
            "for and produces crude oil and natural gas, refines, markets, and distributes "
            "transportation fuels and lubricants, manufactures and sells petrochemical products, "
            "generates power, and transmits natural gas. Chevron operates through two segments: "
            "Upstream and Downstream. The Upstream segment explores for and produces crude oil "
            "and natural gas across the United States and internationally. The Downstream segment "
            "manufactures and distributes refined products including gasoline, diesel, and "
            "jet fuels globally. The Company also invests in renewable energy and carbon "
            "capture technologies as part of its lower carbon strategy."
        ),
    },
    {
        "cik": "732834",
        "company": "Visa Inc.",
        "sic": "7374",
        "state_of_inc": "DE",
        "filing_type": "10-K",
        "filing_date": "2023-11-16",
        "period_of_report": "2023-09-30",
        "htm_filing_link": "https://www.sec.gov/Archives/edgar/data/732834/000073283423000003/v-20230930.htm",
        "item_1": (
            "Visa Inc. operates a global payment network that facilitates digital payments "
            "among consumers, merchants, financial institutions, and government entities. "
            "The Company's products and services include consumer credit, debit, and prepaid "
            "products, commercial credit and debit products, and global ATM services. Visa's "
            "payment network connects consumers and businesses in more than 200 countries and "
            "territories. The Company provides transaction processing services through "
            "VisaNet, which can process more than 65,000 transaction messages per second. "
            "Visa also offers value-added services including fraud management tools, dispute "
            "resolution, and data analytics capabilities to its financial institution clients."
        ),
    },
    {
        "cik": "14272",
        "company": "The Procter & Gamble Company",
        "sic": "2840",
        "state_of_inc": "OH",
        "filing_type": "10-K",
        "filing_date": "2023-08-04",
        "period_of_report": "2023-06-30",
        "htm_filing_link": "https://www.sec.gov/Archives/edgar/data/14272/000001427223000024/pg-20230630.htm",
        "item_1": (
            "The Procter & Gamble Company provides branded consumer packaged goods worldwide. "
            "The Company sells its products in more than 180 countries and territories. "
            "Products are sold through mass merchandisers, grocery stores, membership club "
            "stores, drug stores, department stores, specialty beauty stores, high-frequency "
            "stores, and e-commerce. The Company operates through five segments: Beauty, "
            "Grooming, Health Care, Fabric & Home Care, and Baby, Feminine & Family Care. "
            "Key brands include Tide, Pampers, Always, Gillette, Oral-B, Crest, Head & "
            "Shoulders, Pantene, Olay, SK-II, Febreze, Dawn, and Bounty. The Company focuses "
            "on product innovation, brand building, and productivity improvement."
        ),
    },
    {
        "cik": "1637774",
        "company": "Mastercard Incorporated",
        "sic": "7374",
        "state_of_inc": "DE",
        "filing_type": "10-K",
        "filing_date": "2024-02-13",
        "period_of_report": "2023-12-31",
        "htm_filing_link": "https://www.sec.gov/Archives/edgar/data/1637774/000163777424000003/ma-20231231.htm",
        "item_1": (
            "Mastercard Incorporated is a technology company in the global payments industry "
            "that connects consumers, financial institutions, merchants, governments, and "
            "businesses. The Company provides payment solutions through its proprietary "
            "network in more than 210 countries and territories. Mastercard offers a range "
            "of payment solutions including credit, debit, prepaid, commercial, and "
            "contactless payment products. The Company generates revenue from providing "
            "payment transaction switching and processing services and from fees for "
            "other payment-related products and services. Mastercard also provides "
            "services including loyalty and reward programs, data analytics, fraud prevention, "
            "and consulting services to its financial institution partners."
        ),
    },
    {
        "cik": "1085869",
        "company": "UnitedHealth Group Incorporated",
        "sic": "6324",
        "state_of_inc": "DE",
        "filing_type": "10-K",
        "filing_date": "2024-02-14",
        "period_of_report": "2023-12-31",
        "htm_filing_link": "https://www.sec.gov/Archives/edgar/data/1085869/000108586924000003/unh-20231231.htm",
        "item_1": (
            "UnitedHealth Group Incorporated operates as a diversified health care company "
            "in the United States. The Company operates through two distinct platforms: "
            "UnitedHealthcare, which provides health benefits, and Optum, which provides "
            "health services. UnitedHealthcare offers consumer-oriented health benefit plans "
            "and services to individuals, employers, and Medicare and Medicaid beneficiaries. "
            "Optum Health provides care delivery and care management services to physicians, "
            "patients, and health plans. Optum Insight provides data, analytics, research, "
            "consulting, technology, and managed services to health systems and insurers. "
            "Optum Rx provides pharmaceutical care services and manages pharmacy benefits."
        ),
    },
]


# ---------------------------------------------------------------------------
# Simple sentence-based description extractor (fallback for offline testing)
# ---------------------------------------------------------------------------

def _simple_extract(text: str) -> dict:
    """
    Extract a 2-4 sentence description from Item 1 text without a model.
    Splits on '. ' followed by a capital letter, but skips common abbreviations.
    Used as a GLiNER2 fallback when the model cannot be downloaded.
    """
    # Temporarily replace known abbreviations so they don't trigger splits
    protected = re.sub(r'\b(Inc|Corp|Co|Ltd|Mr|Mrs|Ms|Dr|vs|et|al|approx)\.',
                        lambda m: m.group(0).replace(".", "<<<DOT>>>"), text.strip())
    # Split using lookbehind so the period stays with each sentence
    parts = re.split(r'(?<=\.)\s+(?=[A-Z])', protected)
    # Restore dots and filter short fragments
    sentences = [p.replace("<<<DOT>>>", ".").strip() for p in parts]
    chosen = [s for s in sentences if len(s) > 30][:3]
    description = " ".join(chosen) if chosen else ""
    return {"company": [{"description": description}]} if description else {"company": []}


def _mock_extractor():
    """Return a mock GLiNER2 extractor backed by the simple sentence extractor."""
    mock = MagicMock()
    mock.batch_extract_json.side_effect = lambda texts, schema: [
        _simple_extract(t) for t in texts
    ]
    mock.extract_json.side_effect = lambda text, schema: _simple_extract(text)
    return mock


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def write_synthetic_filings(base_dir: Path) -> Path:
    """Write synthetic filings to data/smoke/EXTRACTED_FILINGS/."""
    filing_dir = base_dir / "EXTRACTED_FILINGS"
    filing_dir.mkdir(parents=True, exist_ok=True)
    for f in FILINGS:
        path = filing_dir / f"{f['cik']}.json"
        path.write_text(json.dumps(f))
    print(f"  Written {len(FILINGS)} synthetic filings to {filing_dir}")
    return filing_dir


def write_metadata(base_dir: Path) -> Path:
    """Write cik_metadata.jsonl from the synthetic filings."""
    meta_path = base_dir / "cik_metadata.jsonl"
    with open(meta_path, "w") as fh:
        for f in FILINGS:
            fh.write(json.dumps({
                "cik": f["cik"],
                "company_name": f["company"],
                "sic_code": f["sic"],
                "tickers": [],
                "state_of_inc": f["state_of_inc"],
                "state_location": f["state_of_inc"],
            }) + "\n")
    print(f"  Written metadata for {len(FILINGS)} companies to {meta_path}")
    return meta_path


def jsonl_to_csv(jsonl_path: Path, csv_path: Path) -> None:
    """Convert dataset.jsonl to dataset.csv."""
    fieldnames = [
        "cik", "company_name", "sic_code", "nace_code",
        "filing_type", "filing_date", "period_of_report",
        "state_of_inc", "filing_url", "description",
    ]
    with open(jsonl_path) as fin, open(csv_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for line in fin:
            line = line.strip()
            if line:
                writer.writerow(json.loads(line))
    print(f"  CSV written to {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import importlib.util

    root = Path(__file__).parent
    smoke_dir = root / "data" / "smoke"
    jsonl_out = root / "data" / "sample_dataset.jsonl"
    csv_out   = root / "data" / "sample_dataset.csv"

    print("── Step 1: writing synthetic filings ──────────────────────────────")
    filing_dir = write_synthetic_filings(smoke_dir)
    meta_path  = write_metadata(smoke_dir)

    print("\n── Step 2: running build_dataset (offline GLiNER2 fallback) ───────")
    # Import 3_build_dataset via importlib (numeric prefix)
    spec = importlib.util.spec_from_file_location("_dataset", root / "3_build_dataset.py")
    ds = importlib.util.module_from_spec(spec)
    # Patch gliner2 before exec so the import doesn't fail
    import sys
    from unittest.mock import MagicMock
    sys.modules.setdefault("gliner2", MagicMock())
    spec.loader.exec_module(ds)

    mock_ext = _mock_extractor()
    jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    if jsonl_out.exists():
        jsonl_out.unlink()

    with patch.object(ds, "_get_extractor", return_value=mock_ext):
        ds.build_dataset(
            extracted_dir=filing_dir,
            metadata_path=meta_path,
            nace_map_path=None,
            output_path=jsonl_out,
            batch_size=8,
        )

    print("\n── Step 3: converting to CSV ───────────────────────────────────────")
    jsonl_to_csv(jsonl_out, csv_out)

    # Quick sanity check
    with open(jsonl_out) as f:
        records = [json.loads(l) for l in f if l.strip()]
    print(f"\n  Records: {len(records)}/{len(FILINGS)}")
    print(f"  Sample:  [{records[0]['company_name']}] {records[0]['description'][:80]}…")
    print(f"\n  Output JSONL: {jsonl_out}")
    print(f"  Output CSV:   {csv_out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
