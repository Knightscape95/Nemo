import unittest
import re
from pathlib import Path

from standalone_nemo_solver import (
    BitRow,
    GravityRow,
    NumeralRow,
    SymbolRow,
    TextRow,
    UnitRow,
    _solve_bit_local,
    _solve_gravity_local,
    _solve_numeral_local,
    _solve_symbol_local,
    _solve_text_local,
    _solve_unit_local,
    build_solver,
    load_rows,
    parse_bit_row,
    parse_gravity_row,
    parse_numeral_row,
    parse_symbol_row,
    parse_text_row,
    parse_unit_row,
    route_row,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = str((REPO_ROOT / "train.csv").resolve())
SOLVER_PATH = (REPO_ROOT / "standalone_nemo_solver.py").resolve()


class TestStandaloneNemoSolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = load_rows(CSV_PATH)
        cls.solver = build_solver(CSV_PATH)

    def test_routing_counts(self) -> None:
        counts = {"bit": 0, "text": 0, "symbol": 0, "numeral": 0, "gravity": 0, "unit": 0}
        for raw in self.rows:
            domain, _ = route_row(raw)
            if domain is not None:
                counts[domain] += 1
        self.assertEqual(counts["bit"], 1602)
        self.assertEqual(counts["text"], 1576)
        self.assertEqual(counts["symbol"], 1555)
        self.assertEqual(counts["numeral"], 1576)
        self.assertEqual(counts["gravity"], 1597)
        self.assertEqual(counts["unit"], 1594)

    def test_known_row_parsing(self) -> None:
        bit_target = next(row for row in self.rows if row.row_id == "00066667")
        bit_row = parse_bit_row(bit_target)
        self.assertIsNotNone(bit_row)
        assert bit_row is not None
        self.assertEqual(format(bit_row.query_input, "08b"), "00110100")

        text_target = next(row for row in self.rows if row.row_id == "00189f6a")
        text_row = parse_text_row(text_target)
        self.assertIsNotNone(text_row)
        assert text_row is not None
        self.assertEqual(text_row.query_cipher, "trb wzrswvog hffk")

        symbol_target = next(row for row in self.rows if row.row_id == "00457d26")
        symbol_row = parse_symbol_row(symbol_target)
        self.assertIsNotNone(symbol_row)
        assert symbol_row is not None
        self.assertEqual(symbol_row.query_expr, "[[-!'")

        numeral_target = next(row for row in self.rows if row.row_id == "001b24c4")
        numeral_row = parse_numeral_row(numeral_target)
        self.assertIsNotNone(numeral_row)
        assert numeral_row is not None
        self.assertEqual(numeral_row.query_number, 38)

        gravity_target = next(row for row in self.rows if row.row_id == "0040ff76")
        gravity_row = parse_gravity_row(gravity_target)
        self.assertIsNotNone(gravity_row)
        assert gravity_row is not None
        self.assertAlmostEqual(gravity_row.query_time, 4.41)

        unit_target = next(row for row in self.rows if row.row_id == "00208201")
        unit_row = parse_unit_row(unit_target)
        self.assertIsNotNone(unit_row)
        assert unit_row is not None
        self.assertAlmostEqual(unit_row.query_value, 25.09)

    def test_standalone_file_is_independent(self) -> None:
        source = SOLVER_PATH.read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"^\s*(from|import)\s+bit_manipulation_decoder\b", source, re.M))
        self.assertIsNone(re.search(r"^\s*(from|import)\s+text_cipher_decoder\b", source, re.M))
        self.assertIsNone(re.search(r"^\s*(from|import)\s+symbol_transform_decoder\b", source, re.M))
        self.assertNotIn("fallback", source)
        self.assertNotIn("oracle", source)

    def test_local_solvers_work_without_training_lookup(self) -> None:
        bit_row = BitRow(
            row_id="local-bit",
            prompt="",
            examples=[
                (0b00000001, 0b00000010),
                (0b00000101, 0b00001010),
                (0b10000000, 0b00000001),
            ],
            query_input=0b01010101,
            answer_output=0b10101010,
        )
        bit_prediction, bit_candidates = _solve_bit_local(bit_row)
        self.assertEqual(bit_prediction, "10101010")
        self.assertGreater(bit_candidates, 0)

        vocab_freq = {"dragon": 1}
        vocab_by_len = {6: ["dragon"]}
        text_row = TextRow(
            row_id="local-text",
            prompt="",
            examples=[("abcdef", "dragon")],
            query_cipher="abcdef",
            answer_plain="dragon",
        )
        text_prediction, text_candidates = _solve_text_local(text_row, vocab_by_len, vocab_freq)
        self.assertEqual(text_prediction, "dragon")
        self.assertGreater(text_candidates, 0)

        symbol_row = SymbolRow(
            row_id="local-symbol",
            prompt="",
            examples=[("ab+cd", "ab"), ("xy-zq", "xy")],
            query_expr="mn*op",
            answer_text="mn",
            family="symbol_string",
        )
        symbol_prediction, symbol_candidates = _solve_symbol_local(symbol_row)
        self.assertEqual(symbol_prediction, "mn")
        self.assertGreater(symbol_candidates, 0)

        numeral_row = NumeralRow(
            row_id="local-numeral",
            prompt="",
            examples=[(4, "IV"), (9, "IX"), (14, "XIV")],
            query_number=38,
            answer_text="XXXVIII",
        )
        numeral_prediction, numeral_candidates = _solve_numeral_local(numeral_row)
        self.assertEqual(numeral_prediction, "XXXVIII")
        self.assertGreater(numeral_candidates, 0)

        gravity_row = GravityRow(
            row_id="local-gravity",
            prompt="",
            examples=[(1.0, 5.0), (2.0, 20.0), (3.0, 45.0)],
            query_time=4.0,
            answer_text="80.0",
        )
        gravity_prediction, gravity_candidates = _solve_gravity_local(gravity_row)
        self.assertEqual(gravity_prediction, "80.0")
        self.assertGreater(gravity_candidates, 0)

        unit_row = UnitRow(
            row_id="local-unit",
            prompt="",
            examples=[(10.0, 6.00), (20.0, 11.00), (30.0, 16.00)],
            query_value=25.0,
            answer_text="13.50",
        )
        unit_prediction, unit_candidates = _solve_unit_local(unit_row)
        self.assertEqual(unit_prediction, "13.50")
        self.assertGreater(unit_candidates, 0)

    def test_full_train_exact_match(self) -> None:
        summary = self.solver.evaluate()
        self.assertEqual(summary.selected_count, 9500)
        self.assertEqual(summary.exact_match, 9500)
        self.assertAlmostEqual(summary.accuracy, 1.0)
        self.assertEqual(summary.by_domain["bit"]["exact"], 1602)
        self.assertEqual(summary.by_domain["text"]["exact"], 1576)
        self.assertEqual(summary.by_domain["symbol"]["exact"], 1555)
        self.assertEqual(summary.by_domain["numeral"]["exact"], 1576)
        self.assertEqual(summary.by_domain["gravity"]["exact"], 1597)
        self.assertEqual(summary.by_domain["unit"]["exact"], 1594)
        self.assertEqual(summary.failures, [])


if __name__ == "__main__":
    unittest.main()
