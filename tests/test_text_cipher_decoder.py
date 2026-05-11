import unittest

from text_cipher_decoder import (
    GeneratorRule,
    infer_generator_rule,
    load_rows,
    parse_text_cipher_row,
    run_evaluation,
    select_text_cipher_rows,
)


CSV_PATH = "/home/runner/work/Nemo/Nemo/train.csv"


class TestTextCipherDecoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_rows = load_rows(CSV_PATH)
        cls.selected_rows, cls.sanity = select_text_cipher_rows(cls.raw_rows)

    def test_selection_rule_counts(self) -> None:
        self.assertEqual(self.sanity["selected_count"], 1576)
        self.assertEqual(self.sanity["decrypt_count"], 1576)
        self.assertGreaterEqual(self.sanity["marker_count"], self.sanity["selected_count"])

    def test_known_row_parsing(self) -> None:
        target = next(r for r in self.raw_rows if r.row_id == "00189f6a")
        parsed = parse_text_cipher_row(target)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.query_cipher, "trb wzrswvog hffk")
        self.assertEqual(parsed.answer_plain, "cat imagines book")
        self.assertGreaterEqual(len(parsed.examples), 3)

    def test_infer_rule_detects_contradiction(self) -> None:
        rule = infer_generator_rule(
            [
                ("ab", "cd"),
                ("a", "e"),
            ]
        )
        self.assertTrue(rule.contradictions)

    def test_generator_rule_decode(self) -> None:
        rule = GeneratorRule(cipher_to_plain={"x": "a", "y": "b"}, plain_to_cipher={"a": "x", "b": "y"}, contradictions=[])
        self.assertEqual(rule.decode_text("xy yx"), "ab ba")

    def test_full_train_text_cipher_accuracy(self) -> None:
        report = run_evaluation(CSV_PATH)
        self.assertEqual(report.selected_count, 1576)
        self.assertEqual(report.exact_match, report.selected_count)
        self.assertAlmostEqual(report.accuracy, 1.0)
        self.assertEqual(len(report.failures), 0)


if __name__ == "__main__":
    unittest.main()
