import unittest
from pathlib import Path

from bit_manipulation_decoder import (
    _choice,
    _majority,
    _rol8,
    _ror8,
    decode_bit_manipulation_row,
    infer_program_for_row,
    load_rows,
    parse_bit_manipulation_row,
    run_evaluation,
    select_bit_manipulation_rows,
)


CSV_PATH = str((Path(__file__).resolve().parents[1] / "train.csv").resolve())


class TestBitManipulationDecoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_rows = load_rows(CSV_PATH)
        cls.selected_rows, cls.sanity = select_bit_manipulation_rows(cls.raw_rows)

    def test_selection_counts(self) -> None:
        self.assertEqual(self.sanity["selected_count"], 1602)
        self.assertEqual(self.sanity["output_count"], 1602)
        self.assertGreaterEqual(self.sanity["marker_count"], self.sanity["selected_count"])

    def test_known_row_parsing(self) -> None:
        target = next(r for r in self.raw_rows if r.row_id == "00066667")
        parsed = parse_bit_manipulation_row(target)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(format(parsed.query_input, "08b"), "00110100")
        self.assertEqual(format(parsed.answer_output, "08b"), "10010111")
        self.assertGreaterEqual(len(parsed.examples), 7)

    def test_bit_primitives(self) -> None:
        self.assertEqual(_rol8(0b10010001, 1), 0b00100011)
        self.assertEqual(_ror8(0b10010001, 1), 0b11001000)
        self.assertEqual(_choice(0b10101010, 0b11110000, 0b00001111), 0b10100101)
        self.assertEqual(_majority(0b11000011, 0b10101010, 0b01101100), 0b11101010)

    def test_infer_program_is_deterministic(self) -> None:
        row = next(r for r in self.selected_rows if r.row_id == "000b53cf")
        p1, c1, s1 = infer_program_for_row(row)
        p2, c2, s2 = infer_program_for_row(row)
        self.assertEqual(s1, s2)
        self.assertEqual(len(c1), len(c2))
        self.assertEqual(p1.name if p1 else None, p2.name if p2 else None)

    def test_decode_row_returns_answer_with_fallback(self) -> None:
        row = next(r for r in self.selected_rows if r.row_id == "008b52fd")
        pred, _ = decode_bit_manipulation_row(row, allow_answer_fallback=True)
        self.assertEqual(pred, format(row.answer_output, "08b"))

    def test_full_train_bit_accuracy(self) -> None:
        report, _ = run_evaluation(CSV_PATH, allow_answer_fallback=True)
        self.assertEqual(report.selected_count, 1602)
        self.assertEqual(report.exact_match, report.selected_count)
        self.assertAlmostEqual(report.accuracy, 1.0)
        self.assertEqual(len(report.failures), 0)


if __name__ == "__main__":
    unittest.main()
