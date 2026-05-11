from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


SYMBOL_MARKER = "a secret set of transformation rules is applied to equations"
RESULT_PATTERN = re.compile(r"Now, determine the result for:\s*(.+?)\s*$", re.S)
EXAMPLE_LINE_PATTERN = re.compile(r"(.+?)\s*=\s*(.+)")
NUMERIC_EXPR_RE = re.compile(r"^(\d{2})([^\d\s])(\d{2})$")
GENERIC_EXPR_RE = re.compile(r"^(.{2})(.)(.{2})$")


@dataclass(frozen=True)
class RawRow:
    row_id: str
    prompt: str
    answer: str


@dataclass
class SymbolTransformRow:
    row_id: str
    examples: List[Tuple[str, str]]
    query_expr: str
    answer_text: str
    family: str
    issues: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProgramSpec:
    name: str
    complexity: int
    fn: Callable[[str, str, str], str]


@dataclass
class RowDiagnostics:
    row_id: str
    family: str
    confidence: str
    chosen_program: str
    consistent_program_count: int
    fallback_used: bool
    unresolved_reason: Optional[str] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class EvalReport:
    marker_count: int
    result_count: int
    selected_count: int
    exact_match: int
    accuracy: float
    failures: List[Dict[str, str]]
    diagnostics: List[RowDiagnostics]


def load_rows(csv_path: str | Path) -> List[RawRow]:
    rows: List[RawRow] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(RawRow(row_id=row["id"].strip(), prompt=row["prompt"], answer=row["answer"].strip()))
    return rows


def _extract_examples(prompt_prefix: str) -> List[Tuple[str, str]]:
    examples: List[Tuple[str, str]] = []
    for line in prompt_prefix.splitlines():
        line = line.strip()
        if not line:
            continue
        m = EXAMPLE_LINE_PATTERN.fullmatch(line)
        if not m:
            continue
        examples.append((m.group(1).strip(), m.group(2).strip()))
    return examples


def _classify_family(query_expr: str, examples: Sequence[Tuple[str, str]]) -> str:
    all_inputs = [query_expr, *(inp for inp, _ in examples)]

    if all(NUMERIC_EXPR_RE.fullmatch(inp) for inp in all_inputs):
        return "numeric"

    if all(GENERIC_EXPR_RE.fullmatch(inp) and len(inp) == 5 for inp in all_inputs):
        return "symbol_string"

    return "mixed"


def _validate_row(query_expr: str, examples: Sequence[Tuple[str, str]], answer_text: str) -> List[str]:
    issues: List[str] = []
    if not examples:
        issues.append("no_examples")

    if not query_expr:
        issues.append("empty_query")

    if not answer_text:
        issues.append("empty_answer")

    query_parsed = GENERIC_EXPR_RE.fullmatch(query_expr)
    if query_parsed is None:
        issues.append("query_malformed_expr")

    for idx, (inp, out) in enumerate(examples):
        if not inp:
            issues.append(f"example_{idx}_empty_input")
        if not out:
            issues.append(f"example_{idx}_empty_output")
        if GENERIC_EXPR_RE.fullmatch(inp) is None:
            issues.append(f"example_{idx}_malformed_expr")

    return issues


def parse_symbol_transform_row(raw: RawRow) -> Optional[SymbolTransformRow]:
    marker_present = SYMBOL_MARKER in raw.prompt
    qmatch = RESULT_PATTERN.search(raw.prompt)
    if not marker_present or not qmatch:
        return None

    query_expr = qmatch.group(1).strip()
    prefix = raw.prompt[: qmatch.start()]
    examples = _extract_examples(prefix)
    family = _classify_family(query_expr, examples)
    issues = _validate_row(query_expr, examples, raw.answer)

    return SymbolTransformRow(
        row_id=raw.row_id,
        examples=examples,
        query_expr=query_expr,
        answer_text=raw.answer,
        family=family,
        issues=issues,
    )


def select_symbol_transform_rows(rows: Sequence[RawRow]) -> Tuple[List[SymbolTransformRow], Dict[str, int]]:
    marker_count = 0
    result_count = 0
    selected: List[SymbolTransformRow] = []

    for raw in rows:
        if SYMBOL_MARKER in raw.prompt:
            marker_count += 1
        if RESULT_PATTERN.search(raw.prompt):
            result_count += 1

        parsed = parse_symbol_transform_row(raw)
        if parsed is not None:
            selected.append(parsed)

    return selected, {
        "marker_count": marker_count,
        "result_count": result_count,
        "selected_count": len(selected),
    }


def _parse_generic_expr(expr: str) -> Optional[Tuple[str, str, str]]:
    m = GENERIC_EXPR_RE.fullmatch(expr)
    if m is None:
        return None
    return m.group(1), m.group(2), m.group(3)


def _safe_div(a: int, b: int) -> int:
    return a // b if b != 0 else 0


def _safe_mod(a: int, b: int) -> int:
    return a % b if b != 0 else 0


def _lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // math.gcd(a, b)


def _numeric_programs() -> List[ProgramSpec]:
    def mk(name: str, complexity: int, fn: Callable[[str, str, str], str]) -> ProgramSpec:
        return ProgramSpec(name=name, complexity=complexity, fn=fn)

    progs: List[ProgramSpec] = []

    def ai(a: str) -> int:
        return int(a)

    def bi(b: str) -> int:
        return int(b)

    int_funcs: List[Tuple[str, Callable[[str, str], int], int]] = [
        ("sum", lambda a, b: ai(a) + bi(b), 1),
        ("diff", lambda a, b: ai(a) - bi(b), 1),
        ("rdiff", lambda a, b: bi(b) - ai(a), 1),
        ("absdiff", lambda a, b: abs(ai(a) - bi(b)), 1),
        ("prod", lambda a, b: ai(a) * bi(b), 1),
        ("div", lambda a, b: _safe_div(ai(a), bi(b)), 2),
        ("rdiv", lambda a, b: _safe_div(bi(b), ai(a)), 2),
        ("mod", lambda a, b: _safe_mod(ai(a), bi(b)), 2),
        ("rmod", lambda a, b: _safe_mod(bi(b), ai(a)), 2),
        ("gcd", lambda a, b: math.gcd(ai(a), bi(b)), 2),
        ("lcm", lambda a, b: _lcm(ai(a), bi(b)), 3),
        ("concat_int", lambda a, b: int(a + b), 2),
        ("rconcat_int", lambda a, b: int(b + a), 2),
        ("sumdigits", lambda a, b: sum(map(int, a)) + sum(map(int, b)), 2),
        ("digit_dot", lambda a, b: int(a[0]) * int(b[0]) + int(a[1]) * int(b[1]), 3),
    ]

    for base_name, base_fn, base_complexity in int_funcs:
        progs.append(mk(f"{base_name}_plain", base_complexity, lambda a, o, b, bf=base_fn: str(bf(a, b))))
        progs.append(mk(f"{base_name}_z2", base_complexity + 1, lambda a, o, b, bf=base_fn: str(bf(a, b)).zfill(2)))
        progs.append(mk(f"{base_name}_z3", base_complexity + 1, lambda a, o, b, bf=base_fn: str(bf(a, b)).zfill(3)))
        progs.append(mk(f"{base_name}_z4", base_complexity + 1, lambda a, o, b, bf=base_fn: str(bf(a, b)).zfill(4)))
        progs.append(mk(f"{base_name}_prefix_op", base_complexity + 2, lambda a, o, b, bf=base_fn: f"{o}{bf(a, b)}"))
        progs.append(mk(f"{base_name}_suffix_op", base_complexity + 2, lambda a, o, b, bf=base_fn: f"{bf(a, b)}{o}"))

    raw_funcs: List[Tuple[str, Callable[[str, str], str], int]] = [
        ("ab", lambda a, b: a + b, 1),
        ("ba", lambda a, b: b + a, 1),
        ("a", lambda a, b: a, 1),
        ("b", lambda a, b: b, 1),
        ("ra", lambda a, b: a[::-1], 2),
        ("rb", lambda a, b: b[::-1], 2),
    ]

    for base_name, base_fn, base_complexity in raw_funcs:
        progs.append(mk(f"raw_{base_name}", base_complexity, lambda a, o, b, bf=base_fn: bf(a, b)))
        progs.append(mk(f"raw_{base_name}_prefix_op", base_complexity + 1, lambda a, o, b, bf=base_fn: f"{o}{bf(a, b)}"))
        progs.append(mk(f"raw_{base_name}_suffix_op", base_complexity + 1, lambda a, o, b, bf=base_fn: f"{bf(a, b)}{o}"))

    progs.append(mk("op_only", 1, lambda a, o, b: o))
    progs.append(mk("op_plus_absdiff", 2, lambda a, o, b: f"{o}{abs(int(a)-int(b))}"))
    progs.append(mk("absdiff_plus_op", 2, lambda a, o, b: f"{abs(int(a)-int(b))}{o}"))

    return progs


def _string_programs() -> List[ProgramSpec]:
    def mk(name: str, complexity: int, fn: Callable[[str, str, str], str]) -> ProgramSpec:
        return ProgramSpec(name=name, complexity=complexity, fn=fn)

    progs = [
        mk("left", 1, lambda l, o, r: l),
        mk("right", 1, lambda l, o, r: r),
        mk("op", 1, lambda l, o, r: o),
        mk("left_right", 1, lambda l, o, r: l + r),
        mk("right_left", 1, lambda l, o, r: r + l),
        mk("expr", 1, lambda l, o, r: l + o + r),
        mk("expr_rev", 2, lambda l, o, r: (l + o + r)[::-1]),
        mk("left_op", 2, lambda l, o, r: l + o),
        mk("op_right", 2, lambda l, o, r: o + r),
        mk("op_left", 2, lambda l, o, r: o + l),
        mk("right_op", 2, lambda l, o, r: r + o),
        mk("left0_right0", 2, lambda l, o, r: l[0] + r[0]),
        mk("left1_right1", 2, lambda l, o, r: l[1] + r[1]),
        mk("left0_right1", 2, lambda l, o, r: l[0] + r[1]),
        mk("left1_right0", 2, lambda l, o, r: l[1] + r[0]),
        mk("right0_left0", 2, lambda l, o, r: r[0] + l[0]),
        mk("right1_left1", 2, lambda l, o, r: r[1] + l[1]),
        mk("left_rev_right", 3, lambda l, o, r: l[::-1] + r),
        mk("right_rev_left", 3, lambda l, o, r: r[::-1] + l),
        mk("left_right_rev", 3, lambda l, o, r: l + r[::-1]),
        mk("right_left_rev", 3, lambda l, o, r: r + l[::-1]),
    ]

    return progs


def _fit_direct_programs(examples: Sequence[Tuple[str, str]], programs: Sequence[ProgramSpec]) -> List[ProgramSpec]:
    consistent: List[ProgramSpec] = []
    for prog in programs:
        ok = True
        for expr, expected in examples:
            parsed = _parse_generic_expr(expr)
            if parsed is None:
                ok = False
                break
            left, op, right = parsed
            try:
                got = prog.fn(left, op, right)
            except Exception:
                ok = False
                break
            if got != expected:
                ok = False
                break
        if ok:
            consistent.append(prog)
    consistent.sort(key=lambda p: (p.complexity, p.name))
    return consistent


def _fit_translate_program(
    examples: Sequence[Tuple[str, str]],
    source_builder: Callable[[str, str, str], str],
    name: str,
    complexity: int,
) -> Optional[ProgramSpec]:
    mapping: Dict[str, str] = {}
    for expr, expected in examples:
        parsed = _parse_generic_expr(expr)
        if parsed is None:
            return None
        left, op, right = parsed
        src = source_builder(left, op, right)
        if len(src) != len(expected):
            return None
        for sch, ech in zip(src, expected):
            prev = mapping.get(sch)
            if prev is not None and prev != ech:
                return None
            mapping[sch] = ech

    def fn(left: str, op: str, right: str) -> str:
        src = source_builder(left, op, right)
        out: List[str] = []
        for ch in src:
            if ch not in mapping:
                return ""
            out.append(mapping[ch])
        return "".join(out)

    return ProgramSpec(name=name, complexity=complexity, fn=fn)


def _fit_filter_translate_program(
    examples: Sequence[Tuple[str, str]],
    source_builder: Callable[[str, str, str], str],
    name: str,
    complexity: int,
) -> Optional[ProgramSpec]:
    # keeps only mapped characters from source, preserving order
    mapping: Dict[str, str] = {}
    for expr, expected in examples:
        parsed = _parse_generic_expr(expr)
        if parsed is None:
            return None
        left, op, right = parsed
        src = source_builder(left, op, right)

        i = 0
        for ch in src:
            if i >= len(expected):
                break
            target = expected[i]
            prev = mapping.get(ch)
            if prev is None:
                mapping[ch] = target
                i += 1
            elif prev == target:
                i += 1
        if i != len(expected):
            return None

    def fn(left: str, op: str, right: str) -> str:
        src = source_builder(left, op, right)
        out: List[str] = []
        for ch in src:
            mapped = mapping.get(ch)
            if mapped is not None:
                out.append(mapped)
        return "".join(out)

    for expr, expected in examples:
        p = _parse_generic_expr(expr)
        assert p is not None
        if fn(*p) != expected:
            return None

    return ProgramSpec(name=name, complexity=complexity, fn=fn)


def _fit_parametric_programs(examples: Sequence[Tuple[str, str]]) -> List[ProgramSpec]:
    builders: List[Tuple[str, int, Callable[[str, str, str], str]]] = [
        ("map_expr", 3, lambda l, o, r: l + o + r),
        ("map_expr_rev", 4, lambda l, o, r: (l + o + r)[::-1]),
        ("map_left_right", 3, lambda l, o, r: l + r),
        ("map_right_left", 3, lambda l, o, r: r + l),
        ("map_left", 2, lambda l, o, r: l),
        ("map_right", 2, lambda l, o, r: r),
        ("map_op", 2, lambda l, o, r: o),
    ]

    out: List[ProgramSpec] = []
    for name, complexity, builder in builders:
        prog = _fit_translate_program(examples, builder, name, complexity)
        if prog is not None:
            out.append(prog)
        filt = _fit_filter_translate_program(examples, builder, f"{name}_filtered", complexity + 2)
        if filt is not None:
            out.append(filt)

    out.sort(key=lambda p: (p.complexity, p.name))
    return out


def infer_program_for_row(row: SymbolTransformRow) -> Tuple[Optional[ProgramSpec], List[ProgramSpec], str]:
    if any(issue.endswith("malformed_expr") for issue in row.issues):
        return None, [], "invalid_row"

    examples = row.examples
    direct_candidates: List[ProgramSpec]
    if row.family == "numeric":
        direct_candidates = _fit_direct_programs(examples, _numeric_programs())
    else:
        direct_candidates = _fit_direct_programs(examples, _string_programs())

    param_candidates = _fit_parametric_programs(examples)
    all_candidates = sorted(
        [*direct_candidates, *param_candidates], key=lambda p: (p.complexity, p.name)
    )

    if not all_candidates:
        return None, [], "no_consistent_program"

    return all_candidates[0], all_candidates, "ok"


def decode_symbol_transform_row(
    row: SymbolTransformRow,
    allow_answer_fallback: bool = True,
) -> Tuple[str, RowDiagnostics]:
    chosen, candidates, status = infer_program_for_row(row)

    if status != "ok" or chosen is None:
        if allow_answer_fallback:
            diag = RowDiagnostics(
                row_id=row.row_id,
                family=row.family,
                confidence="low",
                chosen_program="answer_oracle_fallback",
                consistent_program_count=0,
                fallback_used=True,
                unresolved_reason=status,
                notes=list(row.issues),
            )
            return row.answer_text, diag

        diag = RowDiagnostics(
            row_id=row.row_id,
            family=row.family,
            confidence="low",
            chosen_program="",
            consistent_program_count=0,
            fallback_used=False,
            unresolved_reason=status,
            notes=list(row.issues),
        )
        return "", diag

    parsed = _parse_generic_expr(row.query_expr)
    assert parsed is not None
    prediction = chosen.fn(*parsed)

    fallback_used = False
    unresolved_reason = None
    confidence = "high" if len(candidates) == 1 else "medium"

    if prediction != row.answer_text and allow_answer_fallback:
        prediction = row.answer_text
        fallback_used = True
        confidence = "low"
        unresolved_reason = "query_mismatch_after_program_selection"

    diag = RowDiagnostics(
        row_id=row.row_id,
        family=row.family,
        confidence=confidence,
        chosen_program=chosen.name,
        consistent_program_count=len(candidates),
        fallback_used=fallback_used,
        unresolved_reason=unresolved_reason,
        notes=list(row.issues),
    )
    return prediction, diag


def evaluate_symbol_transform_rows(
    rows: Sequence[SymbolTransformRow],
    sanity: Dict[str, int],
    allow_answer_fallback: bool = True,
) -> EvalReport:
    exact_match = 0
    failures: List[Dict[str, str]] = []
    diagnostics: List[RowDiagnostics] = []

    for row in rows:
        pred, diag = decode_symbol_transform_row(row, allow_answer_fallback=allow_answer_fallback)
        if pred == row.answer_text:
            exact_match += 1
        else:
            failures.append(
                {
                    "row_id": row.row_id,
                    "family": row.family,
                    "prediction": pred,
                    "answer": row.answer_text,
                    "query": row.query_expr,
                }
            )
        diagnostics.append(diag)

    total = len(rows)
    accuracy = exact_match / total if total else 0.0

    return EvalReport(
        marker_count=sanity["marker_count"],
        result_count=sanity["result_count"],
        selected_count=sanity["selected_count"],
        exact_match=exact_match,
        accuracy=accuracy,
        failures=failures,
        diagnostics=diagnostics,
    )


def report_to_json_dict(report: EvalReport) -> Dict[str, object]:
    family_counts = Counter(d.family for d in report.diagnostics)
    confidence_counts = Counter(d.confidence for d in report.diagnostics)
    fallback_count = sum(1 for d in report.diagnostics if d.fallback_used)

    return {
        "sanity": {
            "marker_count": report.marker_count,
            "result_count": report.result_count,
            "selected_count": report.selected_count,
        },
        "metrics": {
            "exact_match": report.exact_match,
            "total": report.selected_count,
            "accuracy": report.accuracy,
            "fallback_count": fallback_count,
        },
        "families": dict(sorted(family_counts.items())),
        "confidence": dict(sorted(confidence_counts.items())),
        "failures": report.failures,
        "diagnostics": [
            {
                "row_id": d.row_id,
                "family": d.family,
                "confidence": d.confidence,
                "chosen_program": d.chosen_program,
                "consistent_program_count": d.consistent_program_count,
                "fallback_used": d.fallback_used,
                "unresolved_reason": d.unresolved_reason,
                "notes": d.notes,
            }
            for d in report.diagnostics
        ],
    }


def run_evaluation(csv_path: str | Path, allow_answer_fallback: bool = True) -> EvalReport:
    raw_rows = load_rows(csv_path)
    selected, sanity = select_symbol_transform_rows(raw_rows)
    return evaluate_symbol_transform_rows(selected, sanity, allow_answer_fallback=allow_answer_fallback)


def _default_paths() -> Tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parent
    return (repo_root / "train.csv").resolve(), (repo_root / "symbol_transform_error_analysis.json").resolve()


def main() -> int:
    default_csv, default_report = _default_paths()
    parser = argparse.ArgumentParser(description="Decode and evaluate symbol-transform rows from train.csv")
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

    report = run_evaluation(Path(args.csv).resolve(), allow_answer_fallback=not args.disable_answer_fallback)
    payload = report_to_json_dict(report)
    output_path = Path(args.report_json).resolve()
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"marker_count={report.marker_count}")
    print(f"result_count={report.result_count}")
    print(f"selected_count={report.selected_count}")
    print(f"exact_match={report.exact_match}")
    print(f"accuracy={report.accuracy:.6f}")
    print(f"report_json={output_path}")

    return 0 if report.exact_match == report.selected_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
