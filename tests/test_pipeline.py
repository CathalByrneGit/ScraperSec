"""
tests/test_pipeline.py

TDD test suite for the NACE dataset pipeline.

RED/GREEN approach:
  RED  — run against the original broken source; all tests fail (SyntaxError/ImportError)
  GREEN — run after fixes are applied; all tests pass

Run with:  pytest tests/ -v
"""

import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Dynamic module loading — handles numeric-prefix filenames
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent


def _import(filename: str, mod_name: str):
    """Import a .py file by path, bypassing the numeric-prefix restriction."""
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cik_mod():
    """Import 1_build_cik_list.py as a module."""
    return _import("1_build_cik_list.py", "cik_list_mod")


@pytest.fixture(scope="module")
def dataset_mod():
    """Import 3_build_dataset.py with gliner2 mocked to avoid model loading."""
    mock_gliner2 = MagicMock()
    with patch.dict("sys.modules", {"gliner2": mock_gliner2, "gliner2.GLiNER2": MagicMock()}):
        mod = _import("3_build_dataset.py", "dataset_mod")
    return mod


# ---------------------------------------------------------------------------
# 1_build_cik_list — sic_in_filter
# ---------------------------------------------------------------------------

class TestSicInFilter:
    def test_no_filter_accepts_any_sic(self, cik_mod):
        assert cik_mod.sic_in_filter("7372", set(), []) is True

    def test_no_filter_accepts_empty_sic(self, cik_mod):
        assert cik_mod.sic_in_filter("", set(), []) is True

    def test_specific_code_match(self, cik_mod):
        assert cik_mod.sic_in_filter("7372", {"7372"}, []) is True

    def test_specific_code_no_match(self, cik_mod):
        assert cik_mod.sic_in_filter("5411", {"7372"}, []) is False

    def test_range_match(self, cik_mod):
        assert cik_mod.sic_in_filter("7372", set(), [(7000, 7999)]) is True

    def test_range_boundary_low(self, cik_mod):
        assert cik_mod.sic_in_filter("7000", set(), [(7000, 7999)]) is True

    def test_range_boundary_high(self, cik_mod):
        assert cik_mod.sic_in_filter("7999", set(), [(7000, 7999)]) is True

    def test_range_no_match(self, cik_mod):
        assert cik_mod.sic_in_filter("5411", set(), [(7000, 7999)]) is False

    def test_invalid_sic_with_filter_returns_false(self, cik_mod):
        assert cik_mod.sic_in_filter("INVALID", {"7372"}, []) is False

    def test_empty_sic_with_filter_returns_false(self, cik_mod):
        assert cik_mod.sic_in_filter("", {"7372"}, []) is False

    def test_multiple_ranges(self, cik_mod):
        ranges = [(2000, 3999), (7000, 8999)]
        assert cik_mod.sic_in_filter("2500", set(), ranges) is True
        assert cik_mod.sic_in_filter("7500", set(), ranges) is True
        assert cik_mod.sic_in_filter("5000", set(), ranges) is False


# ---------------------------------------------------------------------------
# 1_build_cik_list — build_cik_list (integration with in-memory zip)
# ---------------------------------------------------------------------------

class TestBuildCikList:
    def _make_zip(self, entries: list) -> bytes:
        """Create an in-memory submissions.zip with given company entries."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for i, entry in enumerate(entries):
                zf.writestr(f"CIK{i:010d}.json", json.dumps(entry))
        return buf.getvalue()

    def test_filters_by_sic_code(self, cik_mod, tmp_path):
        raw = self._make_zip([
            {"cik": "320193", "name": "Apple", "sic": "3571",
             "tickers": [], "stateOfIncorporation": "CA", "stateOfLocation": "CA"},
            {"cik": "12345",  "name": "Other", "sic": "9999",
             "tickers": [], "stateOfIncorporation": "NY", "stateOfLocation": "NY"},
        ])
        cik_mod.build_cik_list(raw, {"3571"}, [], tmp_path)
        lines = (tmp_path / "cik_list.txt").read_text().strip().splitlines()
        assert lines == ["320193"]

    def test_outputs_metadata_jsonl(self, cik_mod, tmp_path):
        raw = self._make_zip([
            {"cik": "320193", "name": "Apple", "sic": "3571",
             "tickers": ["AAPL"], "stateOfIncorporation": "CA", "stateOfLocation": "CA"},
        ])
        cik_mod.build_cik_list(raw, {"3571"}, [], tmp_path)
        records = [
            json.loads(l)
            for l in (tmp_path / "cik_metadata.jsonl").read_text().splitlines()
        ]
        assert len(records) == 1
        assert records[0]["cik"] == "320193"
        assert records[0]["sic_code"] == "3571"

    def test_strips_leading_zeros_from_cik(self, cik_mod, tmp_path):
        raw = self._make_zip([
            {"cik": "0000320193", "name": "Apple", "sic": "3571",
             "tickers": [], "stateOfIncorporation": "CA", "stateOfLocation": "CA"},
        ])
        cik_mod.build_cik_list(raw, {"3571"}, [], tmp_path)
        lines = (tmp_path / "cik_list.txt").read_text().strip().splitlines()
        assert lines == ["320193"]

    def test_no_filter_accepts_all(self, cik_mod, tmp_path):
        raw = self._make_zip([
            {"cik": "1", "name": "A", "sic": "1111",
             "tickers": [], "stateOfIncorporation": "", "stateOfLocation": ""},
            {"cik": "2", "name": "B", "sic": "9999",
             "tickers": [], "stateOfIncorporation": "", "stateOfLocation": ""},
        ])
        cik_mod.build_cik_list(raw, set(), [], tmp_path)
        lines = (tmp_path / "cik_list.txt").read_text().strip().splitlines()
        assert len(lines) == 2

    def test_range_filter(self, cik_mod, tmp_path):
        raw = self._make_zip([
            {"cik": "1", "name": "Mfg",  "sic": "2500",
             "tickers": [], "stateOfIncorporation": "", "stateOfLocation": ""},
            {"cik": "2", "name": "Tech", "sic": "7372",
             "tickers": [], "stateOfIncorporation": "", "stateOfLocation": ""},
            {"cik": "3", "name": "Fin",  "sic": "6020",
             "tickers": [], "stateOfIncorporation": "", "stateOfLocation": ""},
        ])
        cik_mod.build_cik_list(raw, set(), [(2000, 3999), (7000, 8999)], tmp_path)
        lines = (tmp_path / "cik_list.txt").read_text().strip().splitlines()
        assert set(lines) == {"1", "2"}


# ---------------------------------------------------------------------------
# 3_build_dataset — get_item1_text
# ---------------------------------------------------------------------------

class TestGetItem1Text:
    def test_returns_item_1_when_present(self, dataset_mod):
        text = "x" * 200
        assert dataset_mod.get_item1_text({"item_1": text}) == text

    def test_falls_back_to_part_1_item_1(self, dataset_mod):
        text = "y" * 200
        assert dataset_mod.get_item1_text({"part_1_item_1": text}) == text

    def test_falls_back_to_part_1(self, dataset_mod):
        text = "z" * 200
        assert dataset_mod.get_item1_text({"part_1": text}) == text

    def test_returns_none_when_all_missing(self, dataset_mod):
        assert dataset_mod.get_item1_text({}) is None

    def test_ignores_short_text_under_100_chars(self, dataset_mod):
        assert dataset_mod.get_item1_text({"item_1": "too short"}) is None

    def test_prefers_item_1_over_fallbacks(self, dataset_mod):
        primary  = "a" * 200
        fallback = "b" * 200
        result = dataset_mod.get_item1_text({"item_1": primary, "part_1": fallback})
        assert result == primary

    def test_strips_whitespace_before_length_check(self, dataset_mod):
        # 95 real chars wrapped in whitespace — stripped length < 100 → None
        text = "  " + "a" * 95 + "  "
        assert dataset_mod.get_item1_text({"item_1": text}) is None

    def test_exactly_101_chars_accepted(self, dataset_mod):
        text = "a" * 101
        assert dataset_mod.get_item1_text({"item_1": text}) == text


# ---------------------------------------------------------------------------
# 3_build_dataset — load_metadata
# ---------------------------------------------------------------------------

class TestLoadMetadata:
    def test_loads_valid_jsonl(self, dataset_mod, tmp_path):
        f = tmp_path / "meta.jsonl"
        f.write_text(
            json.dumps({"cik": "1", "sic_code": "3571", "company_name": "Apple"}) + "\n"
            + json.dumps({"cik": "2", "sic_code": "7372", "company_name": "MSFT"}) + "\n"
        )
        meta = dataset_mod.load_metadata(f)
        assert len(meta) == 2
        assert meta["1"]["sic_code"] == "3571"
        assert meta["2"]["company_name"] == "MSFT"

    def test_missing_file_returns_empty_dict(self, dataset_mod, tmp_path):
        meta = dataset_mod.load_metadata(tmp_path / "nonexistent.jsonl")
        assert meta == {}

    def test_skips_malformed_lines(self, dataset_mod, tmp_path):
        f = tmp_path / "meta.jsonl"
        f.write_text(
            "not json\n"
            + json.dumps({"cik": "1", "sic_code": "0000"}) + "\n"
        )
        meta = dataset_mod.load_metadata(f)
        assert len(meta) == 1


# ---------------------------------------------------------------------------
# 3_build_dataset — load_nace_map
# ---------------------------------------------------------------------------

class TestLoadNaceMap:
    def test_loads_valid_json(self, dataset_mod, tmp_path):
        f = tmp_path / "nace.json"
        f.write_text(json.dumps({"3571": "C26.20", "7372": "J62.01"}))
        result = dataset_mod.load_nace_map(f)
        assert result == {"3571": "C26.20", "7372": "J62.01"}

    def test_returns_empty_for_none_path(self, dataset_mod):
        assert dataset_mod.load_nace_map(None) == {}

    def test_returns_empty_for_missing_file(self, dataset_mod, tmp_path):
        assert dataset_mod.load_nace_map(tmp_path / "missing.json") == {}


# ---------------------------------------------------------------------------
# 3_build_dataset — extract_description (GLiNER2 mocked)
# ---------------------------------------------------------------------------

class TestExtractDescription:
    def test_returns_description_from_gliner2(self, dataset_mod):
        mock_extractor = MagicMock()
        mock_extractor.extract_json.return_value = {
            "company": [{"description": "Apple makes iPhones and Macs."}]
        }
        with patch.object(dataset_mod, "_get_extractor", return_value=mock_extractor):
            result = dataset_mod.extract_description("Apple Inc. is a technology company.")
        assert result == "Apple makes iPhones and Macs."

    def test_returns_none_for_empty_string(self, dataset_mod):
        assert dataset_mod.extract_description("") is None

    def test_returns_none_for_whitespace_only(self, dataset_mod):
        assert dataset_mod.extract_description("   ") is None

    def test_returns_none_when_company_list_empty(self, dataset_mod):
        mock_extractor = MagicMock()
        mock_extractor.extract_json.return_value = {"company": []}
        with patch.object(dataset_mod, "_get_extractor", return_value=mock_extractor):
            result = dataset_mod.extract_description("Some text about a company.")
        assert result is None

    def test_returns_none_on_gliner2_exception(self, dataset_mod):
        mock_extractor = MagicMock()
        mock_extractor.extract_json.side_effect = RuntimeError("model error")
        with patch.object(dataset_mod, "_get_extractor", return_value=mock_extractor):
            result = dataset_mod.extract_description("Some text about a company.")
        assert result is None

    def test_truncates_input_to_extraction_chars(self, dataset_mod):
        """GLiNER2 should only see EXTRACTION_CHARS characters."""
        long_text = "x" * 10_000
        captured_lengths = []

        mock_extractor = MagicMock()
        def capture(text, schema):
            captured_lengths.append(len(text))
            return {"company": [{"description": "desc"}]}
        mock_extractor.extract_json.side_effect = capture

        with patch.object(dataset_mod, "_get_extractor", return_value=mock_extractor):
            dataset_mod.extract_description(long_text)

        assert captured_lengths[0] == dataset_mod.EXTRACTION_CHARS

    def test_returns_none_for_blank_description(self, dataset_mod):
        mock_extractor = MagicMock()
        mock_extractor.extract_json.return_value = {
            "company": [{"description": "   "}]
        }
        with patch.object(dataset_mod, "_get_extractor", return_value=mock_extractor):
            result = dataset_mod.extract_description("Some text about a company.")
        assert result is None


# ---------------------------------------------------------------------------
# 3_build_dataset — process_batch
# ---------------------------------------------------------------------------

class TestProcessBatch:
    """process_batch uses batch_extract_json (returns a list, one dict per text)."""

    def _mock_batch(self, descriptions: list):
        """Return a mock extractor whose batch_extract_json yields the given descriptions."""
        mock_extractor = MagicMock()
        mock_extractor.batch_extract_json.return_value = [
            {"company": [{"description": d}]} if d else {"company": []}
            for d in descriptions
        ]
        return mock_extractor

    def test_returns_records_for_valid_batch(self, dataset_mod):
        mock_extractor = self._mock_batch(["A tech company."])
        batch = [
            ("123",
             {"filing_type": "10-K", "filing_date": "2023-01-01",
              "period_of_report": "2022-12-31", "htm_filing_link": "http://x"},
             "Apple is a technology company that makes consumer electronics " * 5),
        ]
        with patch.object(dataset_mod, "_get_extractor", return_value=mock_extractor):
            records = dataset_mod.process_batch(
                batch, {}, {"123": {"sic_code": "3571", "company_name": "Apple"}}
            )
        assert len(records) == 1
        assert records[0]["cik"] == "123"
        assert records[0]["description"] == "A tech company."

    def test_skips_items_with_no_description(self, dataset_mod):
        mock_extractor = self._mock_batch([None])  # empty company list
        batch = [("999", {}, "text " * 30)]
        with patch.object(dataset_mod, "_get_extractor", return_value=mock_extractor):
            records = dataset_mod.process_batch(batch, {}, {})
        assert records == []

    def test_applies_nace_map(self, dataset_mod):
        mock_extractor = self._mock_batch(["desc"])
        batch = [("1", {"sic": "3571"}, "text " * 30)]
        with patch.object(dataset_mod, "_get_extractor", return_value=mock_extractor):
            records = dataset_mod.process_batch(batch, {"3571": "C26.20"}, {})
        assert records[0]["nace_code"] == "C26.20"

    def test_missing_nace_map_entry_gives_none(self, dataset_mod):
        mock_extractor = self._mock_batch(["desc"])
        batch = [("1", {"sic": "9999"}, "text " * 30)]
        with patch.object(dataset_mod, "_get_extractor", return_value=mock_extractor):
            records = dataset_mod.process_batch(batch, {"3571": "C26.20"}, {})
        assert records[0]["nace_code"] is None

    def test_handles_empty_batch(self, dataset_mod):
        records = dataset_mod.process_batch([], {}, {})
        assert records == []

    def test_uses_batch_api_not_single_call(self, dataset_mod):
        """Verify batch_extract_json is called once for the whole batch, not per item."""
        mock_extractor = self._mock_batch(["desc A", "desc B", "desc C"])
        batch = [
            (str(i), {}, f"Company {i} makes things. " * 10)
            for i in range(3)
        ]
        with patch.object(dataset_mod, "_get_extractor", return_value=mock_extractor):
            records = dataset_mod.process_batch(batch, {}, {})
        assert mock_extractor.batch_extract_json.call_count == 1
        assert len(records) == 3

    def test_truncates_texts_to_extraction_chars(self, dataset_mod):
        """Texts longer than EXTRACTION_CHARS are truncated before batch call."""
        captured = []
        mock_extractor = MagicMock()
        def capture(texts, schema):
            captured.extend(texts)
            return [{"company": [{"description": "d"}]}]
        mock_extractor.batch_extract_json.side_effect = capture

        long_text = "x" * 10_000
        with patch.object(dataset_mod, "_get_extractor", return_value=mock_extractor):
            dataset_mod.process_batch([("1", {}, long_text)], {}, {})
        assert len(captured[0]) == dataset_mod.EXTRACTION_CHARS


# ---------------------------------------------------------------------------
# 3_build_dataset — build_dataset (end-to-end with mocked model)
# ---------------------------------------------------------------------------

class TestBuildDataset:
    def _write_filing(self, path: Path, data: dict):
        path.write_text(json.dumps(data))

    def _mock_extractor(self, description: str = "desc"):
        """Return a mock whose batch_extract_json returns one result per text."""
        mock_extractor = MagicMock()
        mock_extractor.batch_extract_json.side_effect = lambda texts, schema: [
            {"company": [{"description": description}]} for _ in texts
        ]
        return mock_extractor

    def test_creates_output_jsonl(self, dataset_mod, tmp_path):
        filing_dir = tmp_path / "filings"
        filing_dir.mkdir()
        self._write_filing(filing_dir / "f1.json", {
            "cik": "320193",
            "item_1": "Apple Inc. designs and manufactures consumer electronics. " * 5,
            "filing_type": "10-K",
            "filing_date": "2023-01-01",
            "period_of_report": "2022-12-31",
            "htm_filing_link": "https://sec.gov/x",
        })

        with patch.object(dataset_mod, "_get_extractor",
                          return_value=self._mock_extractor("Apple makes iPhones.")):
            dataset_mod.build_dataset(
                extracted_dir=filing_dir,
                metadata_path=tmp_path / "nonexistent.jsonl",
                nace_map_path=None,
                output_path=tmp_path / "out.jsonl",
            )

        lines = (tmp_path / "out.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["cik"] == "320193"
        assert record["description"] == "Apple makes iPhones."

    def test_resumes_from_existing_output(self, dataset_mod, tmp_path):
        filing_dir = tmp_path / "filings"
        filing_dir.mkdir()
        for cik, name in [("111", "Corp A"), ("222", "Corp B")]:
            self._write_filing(filing_dir / f"{cik}.json", {
                "cik": cik,
                "item_1": f"{name} makes widgets. " * 10,
                "filing_type": "10-K",
                "filing_date": "2023-01-01",
                "period_of_report": "2022-12-31",
                "htm_filing_link": "",
            })

        output = tmp_path / "out.jsonl"
        # Pre-populate: CIK 111 already done
        output.write_text(json.dumps({"cik": "111", "description": "pre-existing"}) + "\n")

        with patch.object(dataset_mod, "_get_extractor",
                          return_value=self._mock_extractor()):
            dataset_mod.build_dataset(
                extracted_dir=filing_dir,
                metadata_path=tmp_path / "meta.jsonl",
                nace_map_path=None,
                output_path=output,
            )

        lines = output.read_text().strip().splitlines()
        ciks = [json.loads(l)["cik"] for l in lines]
        assert "111" in ciks           # pre-existing preserved
        assert "222" in ciks           # newly added
        assert ciks.count("111") == 1  # not duplicated

    def test_batch_size_respected(self, dataset_mod, tmp_path):
        """All records written regardless of batch_size setting."""
        filing_dir = tmp_path / "filings"
        filing_dir.mkdir()
        for i in range(5):
            self._write_filing(filing_dir / f"f{i}.json", {
                "cik": str(i + 100),
                "item_1": f"Company {i} makes products. " * 10,
                "filing_type": "10-K",
                "filing_date": "2023-01-01",
                "period_of_report": "2022-12-31",
                "htm_filing_link": "",
            })

        with patch.object(dataset_mod, "_get_extractor",
                          return_value=self._mock_extractor()):
            dataset_mod.build_dataset(
                extracted_dir=filing_dir,
                metadata_path=tmp_path / "meta.jsonl",
                nace_map_path=None,
                output_path=tmp_path / "out.jsonl",
                batch_size=2,
            )

        lines = (tmp_path / "out.jsonl").read_text().strip().splitlines()
        assert len(lines) == 5

    def test_batch_calls_match_expected_count(self, dataset_mod, tmp_path):
        """With batch_size=2 and 5 filings, batch_extract_json called 3 times (2+2+1)."""
        filing_dir = tmp_path / "filings"
        filing_dir.mkdir()
        for i in range(5):
            self._write_filing(filing_dir / f"f{i}.json", {
                "cik": str(i + 200),
                "item_1": f"Company {i} makes things. " * 10,
                "filing_type": "10-K",
                "filing_date": "2023-01-01",
                "period_of_report": "2022-12-31",
                "htm_filing_link": "",
            })

        mock_extractor = self._mock_extractor()
        with patch.object(dataset_mod, "_get_extractor", return_value=mock_extractor):
            dataset_mod.build_dataset(
                extracted_dir=filing_dir,
                metadata_path=tmp_path / "meta.jsonl",
                nace_map_path=None,
                output_path=tmp_path / "out.jsonl",
                batch_size=2,
            )
        assert mock_extractor.batch_extract_json.call_count == 3  # ceil(5/2)

    def test_skips_filings_with_no_item1(self, dataset_mod, tmp_path):
        filing_dir = tmp_path / "filings"
        filing_dir.mkdir()
        # Filing with no item_1 content
        self._write_filing(filing_dir / "empty.json", {
            "cik": "999",
            "item_1": "",  # empty
            "filing_type": "10-K",
        })

        mock_extractor = MagicMock()
        with patch.object(dataset_mod, "_get_extractor", return_value=mock_extractor):
            dataset_mod.build_dataset(
                extracted_dir=filing_dir,
                metadata_path=tmp_path / "meta.jsonl",
                nace_map_path=None,
                output_path=tmp_path / "out.jsonl",
            )

        assert not (tmp_path / "out.jsonl").exists() or \
               (tmp_path / "out.jsonl").read_text().strip() == ""
