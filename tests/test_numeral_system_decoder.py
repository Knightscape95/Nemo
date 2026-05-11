import unittest
from pathlib import Path

from numeral_system_decoder import (
    audit_rule_consistency,
    decode_numeral_system_row,
    int_to_canonical_roman,
    load_rows,
    parse_numeral_system_row,
    run_evaluation,
    select_numeral_system_rows,
)


CSV_PATH = str((Path(__file__).resolve().parents[1] / "train.csv").resolve())


class TestNumeralSystemDecoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_rows = load_rows(CSV_PATH)
        cls.selected_rows, cls.sanity = select_numeral_system_rows(cls.raw_rows)

    def test_selection_counts_and_audit(self) -> None:
        self.assertEqual(self.sanity["selected_count"], 1576)
        self.assertEqual(self.sanity["marker_count"], 1576)
        self.assertEqual(self.sanity["query_count"], 1576)
        self.assertEqual(self.sanity["marker_query_overlap_count"], 1576)
        self.assertEqual(self.sanity["marker_only_count"], 0)
        self.assertEqual(self.sanity["query_only_count"], 0)

    def test_known_row_parsing(self) -> None:
        target = next(r for r in self.raw_rows if r.row_id == "001b24c4")
        parsed = parse_numeral_system_row(target)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.query_number, 38)
        self.assertEqual(parsed.answer_text, "XXXVIII")
        self.assertGreaterEqual(len(parsed.examples), 4)

    def test_core_roman_conversion(self) -> None:
        self.assertEqual(int_to_canonical_roman(1), "I")
        self.assertEqual(int_to_canonical_roman(4), "IV")
        self.assertEqual(int_to_canonical_roman(9), "IX")
        self.assertEqual(int_to_canonical_roman(40), "XL")
        self.assertEqual(int_to_canonical_roman(44), "XLIV")
        self.assertEqual(int_to_canonical_roman(90), "XC")
        self.assertEqual(int_to_canonical_roman(94), "XCIV")
        self.assertEqual(int_to_canonical_roman(99), "XCIX")
        self.assertEqual(int_to_canonical_roman(100), "C")

    def test_decode_is_deterministic(self) -> None:
        row = next(r for r in self.selected_rows if r.row_id == "00d9f682")
        pred1, diag1 = decode_numeral_system_row(row)
        pred2, diag2 = decode_numeral_system_row(row)
        self.assertEqual(pred1, pred2)
        self.assertEqual(diag1.chosen_rule, diag2.chosen_rule)
        self.assertEqual(diag1.mismatch_reasons, diag2.mismatch_reasons)

    def test_dataset_rule_consistency(self) -> None:
        audits = audit_rule_consistency(self.selected_rows)
        self.assertEqual(len(audits), 1576)
        self.assertTrue(all(a.is_consistent for a in audits))

    def test_full_train_numeral_accuracy_strict(self) -> None:
        report = run_evaluation(CSV_PATH)
        self.assertEqual(report.selected_count, 1576)
        self.assertEqual(report.exact_match, report.selected_count)
        self.assertAlmostEqual(report.accuracy, 1.0)
        self.assertEqual(len(report.failures), 0)


if __name__ == "__main__":
    unittest.main()
