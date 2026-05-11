import unittest
from pathlib import Path

from standalone_nemo_solver import (
    build_solver,
    load_rows,
    parse_bit_row,
    parse_symbol_row,
    parse_text_row,
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
        counts = {"bit": 0, "text": 0, "symbol": 0}
        for raw in self.rows:
            domain, _ = route_row(raw)
            if domain is not None:
                counts[domain] += 1
        self.assertEqual(counts["bit"], 1602)
        self.assertEqual(counts["text"], 1576)
        self.assertEqual(counts["symbol"], 1555)

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

    def test_standalone_file_is_independent(self) -> None:
        source = SOLVER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("bit_manipulation_decoder", source)
        self.assertNotIn("text_cipher_decoder", source)
        self.assertNotIn("symbol_transform_decoder", source)
        self.assertNotIn("fallback", source)
        self.assertNotIn("oracle", source)

    def test_full_train_exact_match(self) -> None:
        summary = self.solver.evaluate()
        self.assertEqual(summary.selected_count, 4733)
        self.assertEqual(summary.exact_match, 4733)
        self.assertAlmostEqual(summary.accuracy, 1.0)
        self.assertEqual(summary.by_domain["bit"]["exact"], 1602)
        self.assertEqual(summary.by_domain["text"]["exact"], 1576)
        self.assertEqual(summary.by_domain["symbol"]["exact"], 1555)
        self.assertEqual(summary.failures, [])


if __name__ == "__main__":
    unittest.main()
