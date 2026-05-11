from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TEXT_CIPHER_MARKER = "secret encryption rules are used on text"
DECRYPT_PATTERN = re.compile(r"Now, decrypt the following text:\s*(.+?)\s*$", re.S)
VALID_TEXT_RE = re.compile(r"^[a-z ]+$")


@dataclass(frozen=True)
class RawRow:
    row_id: str
    prompt: str
    answer: str


@dataclass
class TextCipherRow:
    row_id: str
    examples: List[Tuple[str, str]]
    query_cipher: str
    answer_plain: str
    issues: List[str] = field(default_factory=list)


@dataclass
class GeneratorRule:
    cipher_to_plain: Dict[str, str]
    plain_to_cipher: Dict[str, str]
    contradictions: List[str]

    def decode_text(self, text: str, unresolved_placeholder: str = "?") -> str:
        out: List[str] = []
        for ch in text:
            if ch == " ":
                out.append(" ")
            else:
                out.append(self.cipher_to_plain.get(ch, unresolved_placeholder))
        return "".join(out)


@dataclass
class RowDiagnostics:
    row_id: str
    fallback_used: bool
    unresolved_before_fallback: int
    unresolved_after_fallback: int
    confidence: str
    failure_bucket: Optional[str] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class EvalReport:
    marker_count: int
    decrypt_count: int
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
            rows.append(
                RawRow(
                    row_id=row["id"].strip(),
                    prompt=row["prompt"],
                    answer=row["answer"].strip(),
                )
            )
    return rows


def _extract_examples(prompt_prefix: str) -> List[Tuple[str, str]]:
    examples: List[Tuple[str, str]] = []
    for line in prompt_prefix.splitlines():
        line = line.strip()
        if "->" not in line:
            continue
        left, right = line.split("->", 1)
        examples.append((left.strip(), right.strip()))
    return examples


def _validate_examples_and_text(
    examples: Sequence[Tuple[str, str]], query_cipher: str, answer_plain: str
) -> List[str]:
    issues: List[str] = []

    if not examples:
        issues.append("no_examples")

    if not VALID_TEXT_RE.fullmatch(query_cipher):
        issues.append("query_invalid_charset")

    if not VALID_TEXT_RE.fullmatch(answer_plain):
        issues.append("answer_invalid_charset")

    for idx, (cipher_text, plain_text) in enumerate(examples):
        if not VALID_TEXT_RE.fullmatch(cipher_text):
            issues.append(f"example_{idx}_cipher_invalid_charset")
        if not VALID_TEXT_RE.fullmatch(plain_text):
            issues.append(f"example_{idx}_plain_invalid_charset")

        if len(cipher_text) != len(plain_text):
            issues.append(f"example_{idx}_length_mismatch")

        if len(cipher_text.split()) != len(plain_text.split()):
            issues.append(f"example_{idx}_token_mismatch")

    return issues


def parse_text_cipher_row(raw: RawRow) -> Optional[TextCipherRow]:
    marker_present = TEXT_CIPHER_MARKER in raw.prompt
    query_match = DECRYPT_PATTERN.search(raw.prompt)

    if not marker_present or not query_match:
        return None

    query_cipher = query_match.group(1).strip()
    prompt_prefix = raw.prompt[: query_match.start()]
    examples = _extract_examples(prompt_prefix)
    issues = _validate_examples_and_text(examples, query_cipher, raw.answer)

    return TextCipherRow(
        row_id=raw.row_id,
        examples=examples,
        query_cipher=query_cipher,
        answer_plain=raw.answer,
        issues=issues,
    )


def select_text_cipher_rows(rows: Sequence[RawRow]) -> Tuple[List[TextCipherRow], Dict[str, int]]:
    marker_count = 0
    decrypt_count = 0
    selected: List[TextCipherRow] = []

    for raw in rows:
        if TEXT_CIPHER_MARKER in raw.prompt:
            marker_count += 1
        if DECRYPT_PATTERN.search(raw.prompt):
            decrypt_count += 1

        parsed = parse_text_cipher_row(raw)
        if parsed is not None:
            selected.append(parsed)

    sanity = {
        "marker_count": marker_count,
        "decrypt_count": decrypt_count,
        "selected_count": len(selected),
    }
    return selected, sanity


def infer_generator_rule(examples: Sequence[Tuple[str, str]]) -> GeneratorRule:
    cipher_to_plain: Dict[str, str] = {}
    plain_to_cipher: Dict[str, str] = {}
    contradictions: List[str] = []

    for ex_idx, (cipher_text, plain_text) in enumerate(examples):
        if len(cipher_text) != len(plain_text):
            contradictions.append(f"example_{ex_idx}_length_mismatch")
            continue

        for pos, (cch, pch) in enumerate(zip(cipher_text, plain_text)):
            if cch == " " or pch == " ":
                if cch != pch:
                    contradictions.append(f"example_{ex_idx}_space_mismatch_at_{pos}")
                continue

            previous_plain = cipher_to_plain.get(cch)
            if previous_plain is not None and previous_plain != pch:
                contradictions.append(
                    f"example_{ex_idx}_cipher_conflict_{cch}_{previous_plain}_{pch}_at_{pos}"
                )
                continue

            previous_cipher = plain_to_cipher.get(pch)
            if previous_cipher is not None and previous_cipher != cch:
                contradictions.append(
                    f"example_{ex_idx}_plain_conflict_{pch}_{previous_cipher}_{cch}_at_{pos}"
                )
                continue

            cipher_to_plain[cch] = pch
            plain_to_cipher[pch] = cch

    return GeneratorRule(
        cipher_to_plain=cipher_to_plain,
        plain_to_cipher=plain_to_cipher,
        contradictions=contradictions,
    )


def unresolved_count(text: str) -> int:
    return text.count("?")


def _word_candidates(
    cipher_word: str,
    vocab_by_len: Dict[int, Sequence[str]],
    vocab_freq: Counter,
    c2p: Dict[str, str],
    p2c: Dict[str, str],
) -> List[str]:
    candidates: List[str] = []
    for plain_word in vocab_by_len.get(len(cipher_word), []):
        local_c2p: Dict[str, str] = {}
        local_p2c: Dict[str, str] = {}
        valid = True

        for cch, pch in zip(cipher_word, plain_word):
            existing_plain = c2p.get(cch)
            if existing_plain is not None and existing_plain != pch:
                valid = False
                break

            existing_cipher = p2c.get(pch)
            if existing_cipher is not None and existing_cipher != cch:
                valid = False
                break

            local_plain = local_c2p.get(cch)
            if local_plain is not None and local_plain != pch:
                valid = False
                break

            local_cipher = local_p2c.get(pch)
            if local_cipher is not None and local_cipher != cch:
                valid = False
                break

            local_c2p[cch] = pch
            local_p2c[pch] = cch

        if valid:
            candidates.append(plain_word)

    candidates.sort(key=lambda w: (-vocab_freq[w], w))
    return candidates


def _extend_mapping(
    cipher_word: str,
    plain_word: str,
    c2p: Dict[str, str],
    p2c: Dict[str, str],
) -> Optional[Tuple[Dict[str, str], Dict[str, str]]]:
    new_c2p = dict(c2p)
    new_p2c = dict(p2c)

    for cch, pch in zip(cipher_word, plain_word):
        prev_plain = new_c2p.get(cch)
        if prev_plain is not None and prev_plain != pch:
            return None

        prev_cipher = new_p2c.get(pch)
        if prev_cipher is not None and prev_cipher != cch:
            return None

        new_c2p[cch] = pch
        new_p2c[pch] = cch

    return new_c2p, new_p2c


def _solve_query_with_fallback(
    query_words: Sequence[str],
    c2p: Dict[str, str],
    p2c: Dict[str, str],
    vocab_by_len: Dict[int, Sequence[str]],
    vocab_freq: Counter,
) -> Optional[Tuple[Dict[str, str], Dict[str, str], List[str]]]:
    unresolved_candidates: List[Tuple[int, str, List[str]]] = []

    for idx, cipher_word in enumerate(query_words):
        decoded = "".join(c2p.get(ch, "?") for ch in cipher_word)
        if "?" not in decoded:
            continue
        candidates = _word_candidates(cipher_word, vocab_by_len, vocab_freq, c2p, p2c)
        unresolved_candidates.append((idx, cipher_word, candidates))

    if not unresolved_candidates:
        return c2p, p2c, []

    unresolved_candidates.sort(key=lambda item: (len(item[2]), -len(set(item[1])), item[0]))
    idx, cipher_word, candidates = unresolved_candidates[0]
    if not candidates:
        return None

    for candidate in candidates:
        extended = _extend_mapping(cipher_word, candidate, c2p, p2c)
        if extended is None:
            continue

        next_c2p, next_p2c = extended
        solved = _solve_query_with_fallback(
            query_words=query_words,
            c2p=next_c2p,
            p2c=next_p2c,
            vocab_by_len=vocab_by_len,
            vocab_freq=vocab_freq,
        )
        if solved is None:
            continue

        final_c2p, final_p2c, chosen_words = solved
        return final_c2p, final_p2c, [candidate] + chosen_words

    return None


def decode_text_cipher_row(
    row: TextCipherRow,
    vocab_by_len: Dict[int, Sequence[str]],
    vocab_freq: Counter,
) -> Tuple[str, GeneratorRule, RowDiagnostics]:
    rule = infer_generator_rule(row.examples)

    if rule.contradictions:
        diagnostics = RowDiagnostics(
            row_id=row.row_id,
            fallback_used=False,
            unresolved_before_fallback=-1,
            unresolved_after_fallback=-1,
            confidence="low",
            failure_bucket="inconsistent_mapping",
            notes=list(rule.contradictions),
        )
        return "", rule, diagnostics

    pre_decode = rule.decode_text(row.query_cipher)
    pre_unresolved = unresolved_count(pre_decode)

    fallback_used = False
    if pre_unresolved > 0:
        fallback_used = True
        solved = _solve_query_with_fallback(
            query_words=row.query_cipher.split(),
            c2p=rule.cipher_to_plain,
            p2c=rule.plain_to_cipher,
            vocab_by_len=vocab_by_len,
            vocab_freq=vocab_freq,
        )
        if solved is not None:
            final_c2p, final_p2c, _ = solved
            rule = GeneratorRule(
                cipher_to_plain=final_c2p,
                plain_to_cipher=final_p2c,
                contradictions=rule.contradictions,
            )

    final_decode = rule.decode_text(row.query_cipher)
    post_unresolved = unresolved_count(final_decode)

    confidence = "high" if post_unresolved == 0 else "medium"
    bucket = "unresolved_chars" if post_unresolved else None
    notes = list(row.issues)

    diagnostics = RowDiagnostics(
        row_id=row.row_id,
        fallback_used=fallback_used,
        unresolved_before_fallback=pre_unresolved,
        unresolved_after_fallback=post_unresolved,
        confidence=confidence,
        failure_bucket=bucket,
        notes=notes,
    )
    return final_decode, rule, diagnostics


def build_example_vocabulary(rows: Iterable[TextCipherRow]) -> Tuple[Counter, Dict[int, List[str]]]:
    vocab_freq: Counter = Counter()
    for row in rows:
        for _, plain_text in row.examples:
            for word in plain_text.split():
                vocab_freq[word] += 1

    by_len: Dict[int, List[str]] = defaultdict(list)
    for word in sorted(vocab_freq.keys()):
        by_len[len(word)].append(word)

    return vocab_freq, by_len


def evaluate_text_cipher_rows(rows: Sequence[TextCipherRow], sanity: Dict[str, int]) -> EvalReport:
    vocab_freq, vocab_by_len = build_example_vocabulary(rows)

    exact_match = 0
    failures: List[Dict[str, str]] = []
    diagnostics: List[RowDiagnostics] = []

    for row in rows:
        prediction, rule, row_diag = decode_text_cipher_row(row, vocab_by_len, vocab_freq)

        if row_diag.failure_bucket == "inconsistent_mapping":
            failures.append(
                {
                    "row_id": row.row_id,
                    "bucket": "inconsistent_mapping",
                    "prediction": prediction,
                    "answer_plain": row.answer_plain,
                    "query_cipher": row.query_cipher,
                }
            )
            diagnostics.append(row_diag)
            continue

        if prediction != row.answer_plain:
            bucket = "unresolved_chars" if "?" in prediction else "wrong_prediction"
            row_diag.failure_bucket = bucket
            failures.append(
                {
                    "row_id": row.row_id,
                    "bucket": bucket,
                    "prediction": prediction,
                    "answer_plain": row.answer_plain,
                    "query_cipher": row.query_cipher,
                }
            )
        else:
            exact_match += 1

        diagnostics.append(row_diag)

    accuracy = exact_match / len(rows) if rows else 0.0

    return EvalReport(
        marker_count=sanity["marker_count"],
        decrypt_count=sanity["decrypt_count"],
        selected_count=sanity["selected_count"],
        exact_match=exact_match,
        accuracy=accuracy,
        failures=failures,
        diagnostics=diagnostics,
    )


def report_to_json_dict(report: EvalReport) -> Dict[str, object]:
    failure_buckets = Counter(f["bucket"] for f in report.failures)

    return {
        "sanity": {
            "marker_count": report.marker_count,
            "decrypt_count": report.decrypt_count,
            "selected_count": report.selected_count,
        },
        "metrics": {
            "exact_match": report.exact_match,
            "total": report.selected_count,
            "accuracy": report.accuracy,
        },
        "failure_buckets": dict(sorted(failure_buckets.items())),
        "failures": report.failures,
        "diagnostics": [
            {
                "row_id": d.row_id,
                "fallback_used": d.fallback_used,
                "unresolved_before_fallback": d.unresolved_before_fallback,
                "unresolved_after_fallback": d.unresolved_after_fallback,
                "confidence": d.confidence,
                "failure_bucket": d.failure_bucket,
                "notes": d.notes,
            }
            for d in report.diagnostics
        ],
    }


def run_evaluation(csv_path: str | Path) -> EvalReport:
    raw_rows = load_rows(csv_path)
    selected_rows, sanity = select_text_cipher_rows(raw_rows)
    return evaluate_text_cipher_rows(selected_rows, sanity)


def _default_paths() -> Tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parent
    csv_path = (repo_root / "train.csv").resolve()
    report_path = (repo_root / "text_cipher_error_analysis.json").resolve()
    return csv_path, report_path


def main() -> int:
    default_csv, default_report = _default_paths()
    parser = argparse.ArgumentParser(description="Decode and evaluate text-cipher rows from train.csv")
    parser.add_argument(
        "--csv",
        default=str(default_csv),
        help="CSV path (resolved to an absolute path)",
    )
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
    print(f"decrypt_count={report.decrypt_count}")
    print(f"selected_count={report.selected_count}")
    print(f"exact_match={report.exact_match}")
    print(f"accuracy={report.accuracy:.6f}")
    print(f"report_json={output_path}")

    return 0 if report.exact_match == report.selected_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
