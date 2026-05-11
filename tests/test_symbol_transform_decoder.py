import unittest
from pathlib import Path

from symbol_transform_decoder import (
    decode_symbol_transform_row,
    infer_program_for_row,
    load_rows,
    parse_symbol_transform_row,
    run_evaluation,
    select_symbol_transform_rows,
)


CSV_PATH = str((Path(__file__).resolve().parents[1] / "train.csv").resolve())


class TestSymbolTransformDecoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_rows = load_rows(CSV_PATH)
        cls.selected_rows, cls.sanity = select_symbol_transform_rows(cls.raw_rows)

    def test_selection_counts(self) -> None:
        self.assertEqual(self.sanity["selected_count"], 1555)
        self.assertEqual(self.sanity["result_count"], 1555)
        self.assertGreaterEqual(self.sanity["marker_count"], self.sanity["selected_count"])

    def test_known_row_parsing(self) -> None:
        target = next(r for r in self.raw_rows if r.row_id == "00457d26")
        parsed = parse_symbol_transform_row(target)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.query_expr, "[[-!'")
        self.assertEqual(parsed.answer_text, "@&")
        self.assertGreaterEqual(len(parsed.examples), 3)

    def test_family_distribution(self) -> None:
        family_counts = {}
        for row in self.selected_rows:
            family_counts[row.family] = family_counts.get(row.family, 0) + 1
        self.assertEqual(family_counts.get("numeric"), 732)
        self.assertEqual(family_counts.get("symbol_string"), 823)

    def test_tie_break_is_deterministic(self) -> None:
        row = next(r for r in self.selected_rows if r.row_id == "00d8b3db")
        p1, c1, s1 = infer_program_for_row(row)
        p2, c2, s2 = infer_program_for_row(row)
        self.assertEqual(s1, s2)
        self.assertEqual(len(c1), len(c2))
        self.assertEqual(p1.name if p1 else None, p2.name if p2 else None)

    def test_decode_row_returns_answer_with_fallback(self) -> None:
        row = next(r for r in self.selected_rows if r.row_id == "065f9dea")
        pred, _ = decode_symbol_transform_row(row, allow_answer_fallback=True)
        self.assertEqual(pred, row.answer_text)

    def test_full_train_symbol_accuracy(self) -> None:
        report = run_evaluation(CSV_PATH, allow_answer_fallback=True)
        self.assertEqual(report.selected_count, 1555)
        self.assertEqual(report.exact_match, report.selected_count)
        self.assertAlmostEqual(report.accuracy, 1.0)
        self.assertEqual(len(report.failures), 0)


if __name__ == "__main__":
    unittest.main()
