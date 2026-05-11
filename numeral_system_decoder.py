from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


NUMERAL_MARKER = "numbers are secretly converted into a different numeral system"
QUERY_PATTERN = re.compile(r"Now, write the number\s+(\d+)\s+in the Wonderland numeral system\.\s*$", re.S)
EXAMPLE_LINE_PATTERN = re.compile(r"(\d+)\s*->\s*([A-Z]+)")
VALID_ROMAN_RE = re.compile(r"^[IVXLCDM]+$")


@dataclass(frozen=True)
class RawRow:
    row_id: str
    prompt: str
    answer: str


@dataclass
class NumeralSystemRow:
    row_id: str
    examples: List[Tuple[int, str]]
    query_number: int
    answer_text: str
    issues: List[str] = field(default_factory=list)


@dataclass
class RowRuleAudit:
    row_id: str
    inconsistent_examples: List[Dict[str, str]] = field(default_factory=list)
    query_mismatch: Optional[Dict[str, str]] = None

    @property
    def is_consistent(self) -> bool:
        return not self.inconsistent_examples and self.query_mismatch is None


@dataclass
class RowDiagnostics:
    row_id: str
    confidence: str
    chosen_rule: str
    example_rule_consistent: bool
    query_rule_consistent: bool
    mismatch_reasons: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class EvalReport:
    marker_count: int
    query_count: int
    marker_query_overlap_count: int
    marker_only_count: int
    query_only_count: int
    selected_count: int
    exact_match: int
    accuracy: float
    failures: List[Dict[str, str]]
    diagnostics: List[RowDiagnostics]
    rule_audits: List[RowRuleAudit]


def int_to_canonical_roman(value: int) -> str:
    if value <= 0:
        raise ValueError("Roman numerals are defined for positive integers")

    table = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]

    n = value
    out: List[str] = []
    for integer, roman in table:
        while n >= integer:
            out.append(roman)
            n -= integer
    return "".join(out)


def load_rows(csv_path: str | Path) -> List[RawRow]:
    rows: List[RawRow] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(RawRow(row_id=row["id"].strip(), prompt=row["prompt"], answer=row["answer"].strip()))
    return rows


def _extract_examples(prompt_prefix: str) -> List[Tuple[int, str]]:
    examples: List[Tuple[int, str]] = []
    for line in prompt_prefix.splitlines():
        line = line.strip()
        if not line:
            continue
        m = EXAMPLE_LINE_PATTERN.fullmatch(line)
        if m is None:
            continue
        examples.append((int(m.group(1)), m.group(2)))
    return examples


def _validate_row(examples: Sequence[Tuple[int, str]], query_number: int, answer_text: str) -> List[str]:
    issues: List[str] = []
    if not examples:
        issues.append("no_examples")
    if query_number <= 0:
        issues.append("query_not_positive")

    if not answer_text:
        issues.append("empty_answer")
    elif VALID_ROMAN_RE.fullmatch(answer_text) is None:
        issues.append("answer_invalid_charset")

    for idx, (number, roman) in enumerate(examples):
        if number <= 0:
            issues.append(f"example_{idx}_number_not_positive")
        if VALID_ROMAN_RE.fullmatch(roman) is None:
            issues.append(f"example_{idx}_roman_invalid_charset")

    return issues


def parse_numeral_system_row(raw: RawRow) -> Optional[NumeralSystemRow]:
    marker_present = NUMERAL_MARKER in raw.prompt
    qmatch = QUERY_PATTERN.search(raw.prompt)
    if not marker_present or qmatch is None:
        return None

    query_number = int(qmatch.group(1))
    prefix = raw.prompt[: qmatch.start()]
    examples = _extract_examples(prefix)
    issues = _validate_row(examples, query_number, raw.answer)

    return NumeralSystemRow(
        row_id=raw.row_id,
        examples=examples,
        query_number=query_number,
        answer_text=raw.answer,
        issues=issues,
    )


def select_numeral_system_rows(rows: Sequence[RawRow]) -> Tuple[List[NumeralSystemRow], Dict[str, int]]:
    marker_count = 0
    query_count = 0
    marker_query_overlap_count = 0
    marker_only_count = 0
    query_only_count = 0
    selected: List[NumeralSystemRow] = []

    for raw in rows:
        marker_present = NUMERAL_MARKER in raw.prompt
        query_present = QUERY_PATTERN.search(raw.prompt) is not None

        marker_count += int(marker_present)
        query_count += int(query_present)

        if marker_present and query_present:
            marker_query_overlap_count += 1
        elif marker_present:
            marker_only_count += 1
        elif query_present:
            query_only_count += 1

        parsed = parse_numeral_system_row(raw)
        if parsed is not None:
            selected.append(parsed)

    return selected, {
        "marker_count": marker_count,
        "query_count": query_count,
        "marker_query_overlap_count": marker_query_overlap_count,
        "marker_only_count": marker_only_count,
        "query_only_count": query_only_count,
        "selected_count": len(selected),
    }


def audit_rule_consistency(rows: Sequence[NumeralSystemRow]) -> List[RowRuleAudit]:
    audits: List[RowRuleAudit] = []

    for row in rows:
        inconsistent_examples: List[Dict[str, str]] = []
        for number, observed in row.examples:
            expected = int_to_canonical_roman(number)
            if observed != expected:
                inconsistent_examples.append(
                    {
                        "number": str(number),
                        "observed": observed,
                        "expected": expected,
                    }
                )

        query_expected = int_to_canonical_roman(row.query_number)
        query_mismatch: Optional[Dict[str, str]] = None
        if row.answer_text != query_expected:
            query_mismatch = {
                "number": str(row.query_number),
                "observed": row.answer_text,
                "expected": query_expected,
            }

        audits.append(
            RowRuleAudit(
                row_id=row.row_id,
                inconsistent_examples=inconsistent_examples,
                query_mismatch=query_mismatch,
            )
        )

    return audits


def decode_numeral_system_row(row: NumeralSystemRow) -> Tuple[str, RowDiagnostics]:
    prediction = int_to_canonical_roman(row.query_number)

    mismatch_reasons: List[str] = []
    example_consistent = True
    for number, observed in row.examples:
        if observed != int_to_canonical_roman(number):
            example_consistent = False
            mismatch_reasons.append(f"example_mismatch_{number}")

    query_consistent = prediction == row.answer_text
    if not query_consistent:
        mismatch_reasons.append("query_answer_mismatch")

    confidence = "high" if example_consistent and query_consistent and not row.issues else "medium"

    diagnostics = RowDiagnostics(
        row_id=row.row_id,
        confidence=confidence,
        chosen_rule="canonical_roman",
        example_rule_consistent=example_consistent,
        query_rule_consistent=query_consistent,
        mismatch_reasons=mismatch_reasons,
        notes=list(row.issues),
    )
    return prediction, diagnostics


def evaluate_numeral_system_rows(rows: Sequence[NumeralSystemRow], sanity: Dict[str, int]) -> EvalReport:
    exact_match = 0
    failures: List[Dict[str, str]] = []
    diagnostics: List[RowDiagnostics] = []

    for row in rows:
        prediction, diag = decode_numeral_system_row(row)
        if prediction == row.answer_text:
            exact_match += 1
        else:
            failures.append(
                {
                    "row_id": row.row_id,
                    "prediction": prediction,
                    "answer": row.answer_text,
                    "query_number": str(row.query_number),
                }
            )
        diagnostics.append(diag)

    rule_audits = audit_rule_consistency(rows)
    total = len(rows)
    accuracy = exact_match / total if total else 0.0

    return EvalReport(
        marker_count=sanity["marker_count"],
        query_count=sanity["query_count"],
        marker_query_overlap_count=sanity["marker_query_overlap_count"],
        marker_only_count=sanity["marker_only_count"],
        query_only_count=sanity["query_only_count"],
        selected_count=sanity["selected_count"],
        exact_match=exact_match,
        accuracy=accuracy,
        failures=failures,
        diagnostics=diagnostics,
        rule_audits=rule_audits,
    )


def report_to_json_dict(report: EvalReport) -> Dict[str, object]:
    consistent_rows = sum(1 for audit in report.rule_audits if audit.is_consistent)
    inconsistent_rows = report.selected_count - consistent_rows
    inconsistent_example_pairs = sum(len(audit.inconsistent_examples) for audit in report.rule_audits)
    inconsistent_query_answers = sum(1 for audit in report.rule_audits if audit.query_mismatch is not None)

    query_numbers = [int(f["query_number"]) for f in report.failures]
    confidence_counts: Dict[str, int] = {}
    for diag in report.diagnostics:
        confidence_counts[diag.confidence] = confidence_counts.get(diag.confidence, 0) + 1

    return {
        "selection_audit": {
            "marker_count": report.marker_count,
            "query_count": report.query_count,
            "marker_query_overlap_count": report.marker_query_overlap_count,
            "marker_only_count": report.marker_only_count,
            "query_only_count": report.query_only_count,
            "selected_count": report.selected_count,
        },
        "metrics": {
            "exact_match": report.exact_match,
            "total": report.selected_count,
            "accuracy": report.accuracy,
        },
        "rule_validation": {
            "rule_profile": "canonical_roman",
            "consistent_rows": consistent_rows,
            "inconsistent_rows": inconsistent_rows,
            "inconsistent_example_pairs": inconsistent_example_pairs,
            "inconsistent_query_answers": inconsistent_query_answers,
            "failed_query_min": min(query_numbers) if query_numbers else None,
            "failed_query_max": max(query_numbers) if query_numbers else None,
        },
        "confidence": dict(sorted(confidence_counts.items())),
        "failures": report.failures,
        "diagnostics": [
            {
                "row_id": d.row_id,
                "confidence": d.confidence,
                "chosen_rule": d.chosen_rule,
                "example_rule_consistent": d.example_rule_consistent,
                "query_rule_consistent": d.query_rule_consistent,
                "mismatch_reasons": d.mismatch_reasons,
                "notes": d.notes,
            }
            for d in report.diagnostics
        ],
        "rule_audits": [
            {
                "row_id": a.row_id,
                "is_consistent": a.is_consistent,
                "inconsistent_examples": a.inconsistent_examples,
                "query_mismatch": a.query_mismatch,
            }
            for a in report.rule_audits
        ],
    }


def run_evaluation(csv_path: str | Path) -> EvalReport:
    raw_rows = load_rows(csv_path)
    selected_rows, sanity = select_numeral_system_rows(raw_rows)
    return evaluate_numeral_system_rows(selected_rows, sanity)


def _default_paths() -> Tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parent
    return (repo_root / "train.csv").resolve(), (repo_root / "numeral_system_diagnostics.json").resolve()


def main() -> int:
    default_csv, default_report = _default_paths()
    parser = argparse.ArgumentParser(description="Decode and evaluate numeral-system rows from train.csv")
    parser.add_argument("--csv", default=str(default_csv), help="CSV path (absolute path is used)")
    parser.add_argument(
        "--report-json",
        default=str(default_report),
        help="Output JSON path for full diagnostics report",
    )
    args = parser.parse_args()

    report = run_evaluation(Path(args.csv).resolve())
    payload = report_to_json_dict(report)
    output_path = Path(args.report_json).resolve()
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"marker_count={report.marker_count}")
    print(f"query_count={report.query_count}")
    print(f"marker_query_overlap_count={report.marker_query_overlap_count}")
    print(f"selected_count={report.selected_count}")
    print(f"exact_match={report.exact_match}")
    print(f"accuracy={report.accuracy:.6f}")
    print(f"report_json={output_path}")

    return 0 if report.exact_match == report.selected_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
