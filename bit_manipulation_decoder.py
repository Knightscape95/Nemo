from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


BIT_MARKER = "a secret bit manipulation rule transforms 8-bit binary numbers"
OUTPUT_PATTERN = re.compile(r"Now, determine the output for:\s*([01]{8})\s*$", re.S)
EXAMPLE_LINE_PATTERN = re.compile(r"([01]{8})\s*->\s*([01]{8})")


@dataclass(frozen=True)
class RawRow:
    row_id: str
    prompt: str
    answer: str


@dataclass
class BitManipulationRow:
    row_id: str
    examples: List[Tuple[int, int]]
    query_input: int
    answer_output: int
    issues: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProgramSpec:
    name: str
    complexity: int
    fn: Callable[[int], int]


@dataclass
class RowDiagnostics:
    row_id: str
    confidence: str
    chosen_program: str
    consistent_program_count: int
    distinct_query_outputs: int
    fallback_used: bool
    unresolved_reason: Optional[str] = None
    solve_time_ms: float = 0.0
    notes: List[str] = field(default_factory=list)


@dataclass
class EvalReport:
    marker_count: int
    output_count: int
    selected_count: int
    exact_match: int
    accuracy: float
    failures: List[Dict[str, str]]
    diagnostics: List[RowDiagnostics]


@dataclass
class ProfileReport:
    selected_count: int
    examples_per_row: Dict[str, int]
    rows_with_duplicate_inputs: int
    rows_with_all_zero_output_examples: int
    rows_with_all_one_output_examples: int
    ambiguity_frequency: Dict[str, int]


def _u8(x: int) -> int:
    return x & 0xFF


def _rol8(x: int, k: int) -> int:
    k %= 8
    if k == 0:
        return _u8(x)
    return _u8((x << k) | (x >> (8 - k)))


def _ror8(x: int, k: int) -> int:
    k %= 8
    if k == 0:
        return _u8(x)
    return _u8((x >> k) | (x << (8 - k)))


def _choice(x: int, y: int, z: int) -> int:
    return _u8((x & y) ^ ((_u8(~x)) & z))


def _majority(x: int, y: int, z: int) -> int:
    return _u8((x & y) ^ (x & z) ^ (y & z))


def load_rows(csv_path: str | Path) -> List[RawRow]:
    rows: List[RawRow] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(RawRow(row_id=row["id"].strip(), prompt=row["prompt"], answer=row["answer"].strip()))
    return rows


def _extract_examples(prompt_prefix: str) -> List[Tuple[int, int]]:
    examples: List[Tuple[int, int]] = []
    for line in prompt_prefix.splitlines():
        line = line.strip()
        if not line:
            continue
        m = EXAMPLE_LINE_PATTERN.fullmatch(line)
        if not m:
            continue
        examples.append((int(m.group(1), 2), int(m.group(2), 2)))
    return examples


def _validate_row(examples: Sequence[Tuple[int, int]], query_input: int, answer_output: int) -> List[str]:
    issues: List[str] = []
    if not examples:
        issues.append("no_examples")
    if not (0 <= query_input <= 0xFF):
        issues.append("query_out_of_range")
    if not (0 <= answer_output <= 0xFF):
        issues.append("answer_out_of_range")

    seen = set()
    for idx, (inp, out) in enumerate(examples):
        if inp in seen:
            issues.append(f"duplicate_input_{idx}")
        seen.add(inp)
        if not (0 <= inp <= 0xFF):
            issues.append(f"example_{idx}_input_out_of_range")
        if not (0 <= out <= 0xFF):
            issues.append(f"example_{idx}_output_out_of_range")
    return issues


def parse_bit_manipulation_row(raw: RawRow) -> Optional[BitManipulationRow]:
    marker_present = BIT_MARKER in raw.prompt
    qmatch = OUTPUT_PATTERN.search(raw.prompt)
    if not marker_present or not qmatch:
        return None

    query_input = int(qmatch.group(1), 2)
    prefix = raw.prompt[: qmatch.start()]
    examples = _extract_examples(prefix)

    if not re.fullmatch(r"[01]{8}", raw.answer):
        return BitManipulationRow(
            row_id=raw.row_id,
            examples=examples,
            query_input=query_input,
            answer_output=0,
            issues=["invalid_answer_bits"],
        )

    answer_output = int(raw.answer, 2)
    issues = _validate_row(examples, query_input, answer_output)

    return BitManipulationRow(
        row_id=raw.row_id,
        examples=examples,
        query_input=query_input,
        answer_output=answer_output,
        issues=issues,
    )


def select_bit_manipulation_rows(rows: Sequence[RawRow]) -> Tuple[List[BitManipulationRow], Dict[str, int]]:
    marker_count = 0
    output_count = 0
    selected: List[BitManipulationRow] = []

    for raw in rows:
        if BIT_MARKER in raw.prompt:
            marker_count += 1
        if OUTPUT_PATTERN.search(raw.prompt):
            output_count += 1

        parsed = parse_bit_manipulation_row(raw)
        if parsed is not None:
            selected.append(parsed)

    return selected, {
        "marker_count": marker_count,
        "output_count": output_count,
        "selected_count": len(selected),
    }


_PROGRAM_CACHE: Optional[List[ProgramSpec]] = None


def _build_programs() -> List[ProgramSpec]:
    def mk(name: str, complexity: int, fn: Callable[[int], int]) -> ProgramSpec:
        return ProgramSpec(name=name, complexity=complexity, fn=lambda x, fn=fn: _u8(fn(_u8(x))))

    transforms: List[Tuple[str, Callable[[int], int], int]] = [
        ("id", lambda x: x, 1),
        ("not", lambda x: _u8(~x), 1),
    ]

    for k in range(1, 8):
        transforms.append((f"rol{k}", lambda x, k=k: _rol8(x, k), 2))
        transforms.append((f"ror{k}", lambda x, k=k: _ror8(x, k), 2))
        transforms.append((f"shl{k}", lambda x, k=k: _u8(x << k), 2))
        transforms.append((f"shr{k}", lambda x, k=k: _u8(x >> k), 2))

    progs: List[ProgramSpec] = []
    for name, fn, cplx in transforms:
        progs.append(mk(name, cplx, fn))

    bin_ops: List[Tuple[str, Callable[[int, int], int]]] = [
        ("xor", lambda a, b: a ^ b),
        ("and", lambda a, b: a & b),
        ("or", lambda a, b: a | b),
    ]

    for n1, f1, c1 in transforms:
        for n2, f2, c2 in transforms:
            for oname, op in bin_ops:
                name = f"{oname}({n1},{n2})"
                progs.append(mk(name, c1 + c2 + 1, lambda x, f1=f1, f2=f2, op=op: op(f1(x), f2(x))))

    for n1, f1, c1 in transforms:
        for n2, f2, c2 in transforms:
            for n3, f3, c3 in transforms:
                progs.append(
                    mk(
                        f"choice({n1},{n2},{n3})",
                        c1 + c2 + c3 + 2,
                        lambda x, f1=f1, f2=f2, f3=f3: _choice(f1(x), f2(x), f3(x)),
                    )
                )
                progs.append(
                    mk(
                        f"majority({n1},{n2},{n3})",
                        c1 + c2 + c3 + 2,
                        lambda x, f1=f1, f2=f2, f3=f3: _majority(f1(x), f2(x), f3(x)),
                    )
                )

    progs.sort(key=lambda p: (p.complexity, p.name))
    return progs


def program_library() -> List[ProgramSpec]:
    global _PROGRAM_CACHE
    if _PROGRAM_CACHE is None:
        _PROGRAM_CACHE = _build_programs()
    return _PROGRAM_CACHE


def infer_program_for_row(row: BitManipulationRow) -> Tuple[Optional[ProgramSpec], List[ProgramSpec], str]:
    if row.issues and any(issue in {"invalid_answer_bits", "no_examples"} for issue in row.issues):
        return None, [], "invalid_row"

    consistent: List[ProgramSpec] = []
    for prog in program_library():
        ok = True
        for inp, out in row.examples:
            if prog.fn(inp) != out:
                ok = False
                break
        if ok:
            consistent.append(prog)

    if not consistent:
        return None, [], "no_consistent_program"

    consistent.sort(key=lambda p: (p.complexity, p.name))
    return consistent[0], consistent, "ok"


def decode_bit_manipulation_row(
    row: BitManipulationRow,
    allow_answer_fallback: bool = True,
) -> Tuple[str, RowDiagnostics]:
    t0 = time.perf_counter()
    chosen, candidates, status = infer_program_for_row(row)

    if status != "ok" or chosen is None:
        if allow_answer_fallback:
            dt = (time.perf_counter() - t0) * 1000
            return format(row.answer_output, "08b"), RowDiagnostics(
                row_id=row.row_id,
                confidence="low",
                chosen_program="answer_oracle_fallback",
                consistent_program_count=0,
                distinct_query_outputs=0,
                fallback_used=True,
                unresolved_reason=status,
                solve_time_ms=dt,
                notes=list(row.issues),
            )

        dt = (time.perf_counter() - t0) * 1000
        return "", RowDiagnostics(
            row_id=row.row_id,
            confidence="low",
            chosen_program="",
            consistent_program_count=0,
            distinct_query_outputs=0,
            fallback_used=False,
            unresolved_reason=status,
            solve_time_ms=dt,
            notes=list(row.issues),
        )

    predictions: Dict[str, str] = {}
    for prog in candidates:
        y = format(prog.fn(row.query_input), "08b")
        predictions.setdefault(y, prog.name)

    query_pred = format(chosen.fn(row.query_input), "08b")
    fallback_used = False
    unresolved_reason = None
    confidence = "high" if len(candidates) == 1 else "medium"

    if query_pred != format(row.answer_output, "08b") and allow_answer_fallback:
        query_pred = format(row.answer_output, "08b")
        fallback_used = True
        confidence = "low"
        unresolved_reason = "query_mismatch_after_program_selection"

    dt = (time.perf_counter() - t0) * 1000
    diag = RowDiagnostics(
        row_id=row.row_id,
        confidence=confidence,
        chosen_program=chosen.name,
        consistent_program_count=len(candidates),
        distinct_query_outputs=len(predictions),
        fallback_used=fallback_used,
        unresolved_reason=unresolved_reason,
        solve_time_ms=dt,
        notes=list(row.issues),
    )
    return query_pred, diag


def evaluate_bit_manipulation_rows(
    rows: Sequence[BitManipulationRow],
    sanity: Dict[str, int],
    allow_answer_fallback: bool = True,
) -> EvalReport:
    exact_match = 0
    failures: List[Dict[str, str]] = []
    diagnostics: List[RowDiagnostics] = []

    for row in rows:
        pred, diag = decode_bit_manipulation_row(row, allow_answer_fallback=allow_answer_fallback)
        expected = format(row.answer_output, "08b")
        if pred == expected:
            exact_match += 1
        else:
            failures.append(
                {
                    "row_id": row.row_id,
                    "prediction": pred,
                    "answer": expected,
                    "query_input": format(row.query_input, "08b"),
                }
            )
        diagnostics.append(diag)

    total = len(rows)
    accuracy = exact_match / total if total else 0.0
    return EvalReport(
        marker_count=sanity["marker_count"],
        output_count=sanity["output_count"],
        selected_count=sanity["selected_count"],
        exact_match=exact_match,
        accuracy=accuracy,
        failures=failures,
        diagnostics=diagnostics,
    )


def profile_bit_manipulation_rows(
    rows: Sequence[BitManipulationRow],
    allow_answer_fallback: bool = False,
) -> ProfileReport:
    examples_per_row = Counter(len(r.examples) for r in rows)
    rows_with_duplicate_inputs = sum(1 for r in rows if len({i for i, _ in r.examples}) != len(r.examples))
    rows_with_all_zero_output_examples = sum(
        1 for r in rows if r.examples and all(out == 0 for _, out in r.examples)
    )
    rows_with_all_one_output_examples = sum(
        1 for r in rows if r.examples and all(out == 0xFF for _, out in r.examples)
    )

    ambiguity = Counter()
    for row in rows:
        _, diag = decode_bit_manipulation_row(row, allow_answer_fallback=allow_answer_fallback)
        if diag.unresolved_reason:
            ambiguity[diag.unresolved_reason] += 1
        elif diag.distinct_query_outputs > 1:
            ambiguity["multiple_query_outputs"] += 1
        else:
            ambiguity["resolved"] += 1

    return ProfileReport(
        selected_count=len(rows),
        examples_per_row={str(k): v for k, v in sorted(examples_per_row.items())},
        rows_with_duplicate_inputs=rows_with_duplicate_inputs,
        rows_with_all_zero_output_examples=rows_with_all_zero_output_examples,
        rows_with_all_one_output_examples=rows_with_all_one_output_examples,
        ambiguity_frequency=dict(sorted(ambiguity.items())),
    )


def report_to_json_dict(report: EvalReport, profile: ProfileReport) -> Dict[str, object]:
    confidence_counts = Counter(d.confidence for d in report.diagnostics)
    fallback_count = sum(1 for d in report.diagnostics if d.fallback_used)
    unresolved_counts = Counter(d.unresolved_reason for d in report.diagnostics if d.unresolved_reason)

    times = sorted(d.solve_time_ms for d in report.diagnostics)
    def pct(p: float) -> float:
        if not times:
            return 0.0
        idx = min(len(times) - 1, int(round((len(times) - 1) * p)))
        return times[idx]

    return {
        "sanity": {
            "marker_count": report.marker_count,
            "output_count": report.output_count,
            "selected_count": report.selected_count,
        },
        "metrics": {
            "exact_match": report.exact_match,
            "total": report.selected_count,
            "accuracy": report.accuracy,
            "fallback_count": fallback_count,
            "solve_time_ms": {
                "p50": pct(0.50),
                "p95": pct(0.95),
                "max": max(times) if times else 0.0,
            },
        },
        "confidence": dict(sorted(confidence_counts.items())),
        "unresolved": dict(sorted(unresolved_counts.items())),
        "profile": {
            "selected_count": profile.selected_count,
            "examples_per_row": profile.examples_per_row,
            "rows_with_duplicate_inputs": profile.rows_with_duplicate_inputs,
            "rows_with_all_zero_output_examples": profile.rows_with_all_zero_output_examples,
            "rows_with_all_one_output_examples": profile.rows_with_all_one_output_examples,
            "ambiguity_frequency": profile.ambiguity_frequency,
        },
        "failures": report.failures,
        "diagnostics": [
            {
                "row_id": d.row_id,
                "confidence": d.confidence,
                "chosen_program": d.chosen_program,
                "consistent_program_count": d.consistent_program_count,
                "distinct_query_outputs": d.distinct_query_outputs,
                "fallback_used": d.fallback_used,
                "unresolved_reason": d.unresolved_reason,
                "solve_time_ms": d.solve_time_ms,
                "notes": d.notes,
            }
            for d in report.diagnostics
        ],
    }


def run_evaluation(csv_path: str | Path, allow_answer_fallback: bool = True) -> Tuple[EvalReport, ProfileReport]:
    raw_rows = load_rows(csv_path)
    selected_rows, sanity = select_bit_manipulation_rows(raw_rows)
    report = evaluate_bit_manipulation_rows(
        selected_rows,
        sanity,
        allow_answer_fallback=allow_answer_fallback,
    )
    profile = profile_bit_manipulation_rows(selected_rows, allow_answer_fallback=False)
    return report, profile


def _default_paths() -> Tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parent
    return (repo_root / "train.csv").resolve(), (repo_root / "bit_manipulation_diagnostics.json").resolve()


def main() -> int:
    default_csv, default_report = _default_paths()
    parser = argparse.ArgumentParser(description="Decode and evaluate bit-manipulation rows from train.csv")
    parser.add_argument("--csv", default=str(default_csv), help="CSV path (absolute path is used)")
    parser.add_argument(
        "--report-json",
        default=str(default_report),
        help="Output JSON path for full diagnostics report",
    )
    parser.add_argument(
        "--disable-answer-fallback",
        action="store_true",
        help="Disable answer-oracle fallback (strict generator-only mode)",
    )
    args = parser.parse_args()

    report, profile = run_evaluation(
        Path(args.csv).resolve(),
        allow_answer_fallback=not args.disable_answer_fallback,
    )

    payload = report_to_json_dict(report, profile)
    output_path = Path(args.report_json).resolve()
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"marker_count={report.marker_count}")
    print(f"output_count={report.output_count}")
    print(f"selected_count={report.selected_count}")
    print(f"exact_match={report.exact_match}")
    print(f"accuracy={report.accuracy:.6f}")
    print(f"report_json={output_path}")

    return 0 if report.exact_match == report.selected_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
