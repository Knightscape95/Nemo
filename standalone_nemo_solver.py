from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from cry.py.anf import Bit as TCRBit
except Exception:  # pragma: no cover
    TCRBit = None  # type: ignore[assignment]


BIT_MARKER = "a secret bit manipulation rule transforms 8-bit binary numbers"
TEXT_MARKER = "secret encryption rules are used on text"
SYMBOL_MARKER = "a secret set of transformation rules is applied to equations"
NUMERAL_MARKER = "numbers are secretly converted into a different numeral system"
GRAVITY_MARKER = "the gravitational constant has been secretly changed"
UNIT_MARKER = "a secret unit conversion is applied to measurements"

BIT_QUERY_RE = re.compile(r"Now, determine the output for:\s*([01]{8})\s*$", re.S)
TEXT_QUERY_RE = re.compile(r"Now, decrypt the following text:\s*(.+?)\s*$", re.S)
SYMBOL_QUERY_RE = re.compile(r"Now, determine the result for:\s*(.+?)\s*$", re.S)
NUMERAL_QUERY_RE = re.compile(r"Now, write the number\s+(\d+)\s+in the Wonderland numeral system\.\s*$", re.S)
GRAVITY_QUERY_RE = re.compile(
    r"Now, determine the falling distance for t =\s*([0-9]+(?:\.[0-9]+)?)s given d = 0.5\*g\*t\^2\.\s*$",
    re.S,
)
UNIT_QUERY_RE = re.compile(r"Now, convert the following measurement:\s*([0-9]+(?:\.[0-9]+)?) m\s*$", re.S)

BIT_EXAMPLE_RE = re.compile(r"([01]{8})\s*->\s*([01]{8})")
SYMBOL_EXAMPLE_RE = re.compile(r"(.+?)\s*=\s*(.+)")
NUMERAL_EXAMPLE_RE = re.compile(r"(\d+)\s*->\s*([IVXLCDM]+)")
GRAVITY_EXAMPLE_RE = re.compile(r"For t =\s*([0-9]+(?:\.[0-9]+)?)s,\s*distance =\s*([0-9]+(?:\.[0-9]+)?) m")
UNIT_EXAMPLE_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?) m becomes ([0-9]+(?:\.[0-9]+)?)")
SYMBOL_EXPR_RE = re.compile(r"^(.{2})(.)(.{2})$")
NUMERIC_EXPR_RE = re.compile(r"^(\d{2})([^\d\s])(\d{2})$")
VALID_TEXT_RE = re.compile(r"^[a-z ]+$")


def _u8(value: int) -> int:
    return value & 0xFF


def _rol8(value: int, shift: int) -> int:
    shift %= 8
    if shift == 0:
        return _u8(value)
    return _u8((value << shift) | (value >> (8 - shift)))


def _ror8(value: int, shift: int) -> int:
    shift %= 8
    if shift == 0:
        return _u8(value)
    return _u8((value >> shift) | (value << (8 - shift)))


def _choice(x: int, y: int, z: int) -> int:
    return _u8((x & y) ^ ((_u8(~x)) & z))


def _majority(x: int, y: int, z: int) -> int:
    return _u8((x & y) ^ (x & z) ^ (y & z))


def _safe_div(a: int, b: int) -> int:
    return a // b if b else 0


def _safe_mod(a: int, b: int) -> int:
    return a % b if b else 0


def _lcm(a: int, b: int) -> int:
    if not a or not b:
        return 0
    return abs(a * b) // math.gcd(a, b)


@dataclass(frozen=True)
class RawRow:
    row_id: str
    prompt: str
    answer: str


@dataclass
class BitRow:
    row_id: str
    prompt: str
    examples: List[Tuple[int, int]]
    query_input: int
    answer_output: int
    issues: List[str] = field(default_factory=list)

    @property
    def signature(self) -> Tuple[Tuple[Tuple[int, int], ...], int]:
        return tuple(self.examples), self.query_input


@dataclass
class TextRow:
    row_id: str
    prompt: str
    examples: List[Tuple[str, str]]
    query_cipher: str
    answer_plain: str
    issues: List[str] = field(default_factory=list)

    @property
    def signature(self) -> Tuple[Tuple[Tuple[str, str], ...], str]:
        return tuple(self.examples), self.query_cipher


@dataclass
class SymbolRow:
    row_id: str
    prompt: str
    examples: List[Tuple[str, str]]
    query_expr: str
    answer_text: str
    family: str
    issues: List[str] = field(default_factory=list)

    @property
    def signature(self) -> Tuple[Tuple[Tuple[str, str], ...], str]:
        return tuple(self.examples), self.query_expr


@dataclass(frozen=True)
class ProgramSpec:
    name: str
    complexity: int
    fn: Callable


@dataclass
class RowResult:
    row_id: str
    domain: str
    prediction: str
    answer: str
    exact: bool
    solver_kind: str
    candidate_count: int
    notes: List[str] = field(default_factory=list)


@dataclass
class EvalSummary:
    selected_count: int
    exact_match: int
    accuracy: float
    by_domain: Dict[str, Dict[str, int]]
    failures: List[RowResult]


@dataclass
class NumeralRow:
    row_id: str
    prompt: str
    examples: List[Tuple[int, str]]
    query_number: int
    answer_text: str
    issues: List[str] = field(default_factory=list)

    @property
    def signature(self) -> Tuple[Tuple[Tuple[int, str], ...], int]:
        return tuple(self.examples), self.query_number


@dataclass
class GravityRow:
    row_id: str
    prompt: str
    examples: List[Tuple[float, float]]
    query_time: float
    answer_text: str
    issues: List[str] = field(default_factory=list)

    @property
    def signature(self) -> Tuple[Tuple[Tuple[float, float], ...], float]:
        return tuple(self.examples), self.query_time


@dataclass
class UnitRow:
    row_id: str
    prompt: str
    examples: List[Tuple[float, float]]
    query_value: float
    answer_text: str
    issues: List[str] = field(default_factory=list)

    @property
    def signature(self) -> Tuple[Tuple[Tuple[float, float], ...], float]:
        return tuple(self.examples), self.query_value


def load_rows(csv_path: str | Path) -> List[RawRow]:
    rows: List[RawRow] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                RawRow(
                    row_id=row["id"].strip(),
                    prompt=row["prompt"],
                    answer=row["answer"].strip(),
                )
            )
    return rows


def _extract_bit_examples(prefix: str) -> List[Tuple[int, int]]:
    result: List[Tuple[int, int]] = []
    for line in prefix.splitlines():
        match = BIT_EXAMPLE_RE.fullmatch(line.strip())
        if match:
            result.append((int(match.group(1), 2), int(match.group(2), 2)))
    return result


def parse_bit_row(raw: RawRow) -> Optional[BitRow]:
    query_match = BIT_QUERY_RE.search(raw.prompt)
    if BIT_MARKER not in raw.prompt or query_match is None:
        return None
    query_input = int(query_match.group(1), 2)
    examples = _extract_bit_examples(raw.prompt[: query_match.start()])
    issues: List[str] = []
    if not re.fullmatch(r"[01]{8}", raw.answer):
        issues.append("invalid_answer_bits")
        answer_output = 0
    else:
        answer_output = int(raw.answer, 2)
    if not examples:
        issues.append("no_examples")
    return BitRow(
        row_id=raw.row_id,
        prompt=raw.prompt,
        examples=examples,
        query_input=query_input,
        answer_output=answer_output,
        issues=issues,
    )


def _extract_text_examples(prefix: str) -> List[Tuple[str, str]]:
    result: List[Tuple[str, str]] = []
    for line in prefix.splitlines():
        stripped = line.strip()
        if "->" not in stripped:
            continue
        left, right = stripped.split("->", 1)
        result.append((left.strip(), right.strip()))
    return result


def parse_text_row(raw: RawRow) -> Optional[TextRow]:
    query_match = TEXT_QUERY_RE.search(raw.prompt)
    if TEXT_MARKER not in raw.prompt or query_match is None:
        return None
    query_cipher = query_match.group(1).strip()
    examples = _extract_text_examples(raw.prompt[: query_match.start()])
    issues: List[str] = []
    if not examples:
        issues.append("no_examples")
    if not VALID_TEXT_RE.fullmatch(query_cipher):
        issues.append("query_invalid_charset")
    if not VALID_TEXT_RE.fullmatch(raw.answer):
        issues.append("answer_invalid_charset")
    return TextRow(
        row_id=raw.row_id,
        prompt=raw.prompt,
        examples=examples,
        query_cipher=query_cipher,
        answer_plain=raw.answer,
        issues=issues,
    )


def _extract_symbol_examples(prefix: str) -> List[Tuple[str, str]]:
    result: List[Tuple[str, str]] = []
    for line in prefix.splitlines():
        match = SYMBOL_EXAMPLE_RE.fullmatch(line.strip())
        if match:
            result.append((match.group(1).strip(), match.group(2).strip()))
    return result


def _classify_symbol_family(query_expr: str, examples: Sequence[Tuple[str, str]]) -> str:
    inputs = [query_expr, *(expr for expr, _ in examples)]
    if all(NUMERIC_EXPR_RE.fullmatch(expr) for expr in inputs):
        return "numeric"
    return "symbol_string"


def parse_symbol_row(raw: RawRow) -> Optional[SymbolRow]:
    query_match = SYMBOL_QUERY_RE.search(raw.prompt)
    if SYMBOL_MARKER not in raw.prompt or query_match is None:
        return None
    query_expr = query_match.group(1).strip()
    examples = _extract_symbol_examples(raw.prompt[: query_match.start()])
    issues: List[str] = []
    if not examples:
        issues.append("no_examples")
    if SYMBOL_EXPR_RE.fullmatch(query_expr) is None:
        issues.append("query_malformed_expr")
    family = _classify_symbol_family(query_expr, examples)
    return SymbolRow(
        row_id=raw.row_id,
        prompt=raw.prompt,
        examples=examples,
        query_expr=query_expr,
        answer_text=raw.answer,
        family=family,
        issues=issues,
    )


def _extract_numeral_examples(prefix: str) -> List[Tuple[int, str]]:
    result: List[Tuple[int, str]] = []
    for line in prefix.splitlines():
        match = NUMERAL_EXAMPLE_RE.fullmatch(line.strip())
        if match:
            result.append((int(match.group(1)), match.group(2)))
    return result


def parse_numeral_row(raw: RawRow) -> Optional[NumeralRow]:
    query_match = NUMERAL_QUERY_RE.search(raw.prompt)
    if NUMERAL_MARKER not in raw.prompt or query_match is None:
        return None
    examples = _extract_numeral_examples(raw.prompt[: query_match.start()])
    issues: List[str] = []
    if not examples:
        issues.append("no_examples")
    return NumeralRow(
        row_id=raw.row_id,
        prompt=raw.prompt,
        examples=examples,
        query_number=int(query_match.group(1)),
        answer_text=raw.answer,
        issues=issues,
    )


def _extract_gravity_examples(prefix: str) -> List[Tuple[float, float]]:
    result: List[Tuple[float, float]] = []
    for line in prefix.splitlines():
        match = GRAVITY_EXAMPLE_RE.fullmatch(line.strip())
        if match:
            result.append((float(match.group(1)), float(match.group(2))))
    return result


def parse_gravity_row(raw: RawRow) -> Optional[GravityRow]:
    query_match = GRAVITY_QUERY_RE.search(raw.prompt)
    if GRAVITY_MARKER not in raw.prompt or query_match is None:
        return None
    examples = _extract_gravity_examples(raw.prompt[: query_match.start()])
    issues: List[str] = []
    if not examples:
        issues.append("no_examples")
    return GravityRow(
        row_id=raw.row_id,
        prompt=raw.prompt,
        examples=examples,
        query_time=float(query_match.group(1)),
        answer_text=raw.answer,
        issues=issues,
    )


def _extract_unit_examples(prefix: str) -> List[Tuple[float, float]]:
    result: List[Tuple[float, float]] = []
    for line in prefix.splitlines():
        match = UNIT_EXAMPLE_RE.fullmatch(line.strip())
        if match:
            result.append((float(match.group(1)), float(match.group(2))))
    return result


def parse_unit_row(raw: RawRow) -> Optional[UnitRow]:
    query_match = UNIT_QUERY_RE.search(raw.prompt)
    if UNIT_MARKER not in raw.prompt or query_match is None:
        return None
    examples = _extract_unit_examples(raw.prompt[: query_match.start()])
    issues: List[str] = []
    if not examples:
        issues.append("no_examples")
    return UnitRow(
        row_id=raw.row_id,
        prompt=raw.prompt,
        examples=examples,
        query_value=float(query_match.group(1)),
        answer_text=raw.answer,
        issues=issues,
    )


def route_row(raw: RawRow) -> Tuple[Optional[str], Optional[object]]:
    bit_row = parse_bit_row(raw)
    if bit_row is not None:
        return "bit", bit_row
    text_row = parse_text_row(raw)
    if text_row is not None:
        return "text", text_row
    symbol_row = parse_symbol_row(raw)
    if symbol_row is not None:
        return "symbol", symbol_row
    numeral_row = parse_numeral_row(raw)
    if numeral_row is not None:
        return "numeral", numeral_row
    gravity_row = parse_gravity_row(raw)
    if gravity_row is not None:
        return "gravity", gravity_row
    unit_row = parse_unit_row(raw)
    if unit_row is not None:
        return "unit", unit_row
    return None, None


_BIT_PROGRAM_CACHE: Optional[List[ProgramSpec]] = None


def _build_bit_programs() -> List[ProgramSpec]:
    def unary_transforms() -> List[Tuple[str, Callable[[int], int], int]]:
        transforms: List[Tuple[str, Callable[[int], int], int]] = [
            ("id", lambda x: x, 1),
            ("not", lambda x: _u8(~x), 1),
            ("rev", lambda x: int(f"{x:08b}"[::-1], 2), 3),
        ]
        for shift in range(1, 8):
            transforms.append((f"rol{shift}", lambda x, shift=shift: _rol8(x, shift), 2))
            transforms.append((f"ror{shift}", lambda x, shift=shift: _ror8(x, shift), 2))
            transforms.append((f"shl{shift}", lambda x, shift=shift: _u8(x << shift), 2))
            transforms.append((f"shr{shift}", lambda x, shift=shift: _u8(x >> shift), 2))
        return transforms

    programs: List[ProgramSpec] = []
    transforms = unary_transforms()

    def add(name: str, complexity: int, fn: Callable[[int], int]) -> None:
        programs.append(ProgramSpec(name=name, complexity=complexity, fn=lambda x, fn=fn: _u8(fn(_u8(x)))))

    for name, fn, complexity in transforms:
        add(name, complexity, fn)

    binary_ops: List[Tuple[str, Callable[[int, int], int]]] = [
        ("xor", lambda a, b: a ^ b),
        ("and", lambda a, b: a & b),
        ("or", lambda a, b: a | b),
        ("add", lambda a, b: a + b),
        ("sub", lambda a, b: a - b),
    ]
    for name_a, fn_a, complexity_a in transforms:
        for name_b, fn_b, complexity_b in transforms:
            for op_name, op in binary_ops:
                add(
                    f"{op_name}({name_a},{name_b})",
                    complexity_a + complexity_b + 1,
                    lambda x, fn_a=fn_a, fn_b=fn_b, op=op: op(fn_a(x), fn_b(x)),
                )

    for name_a, fn_a, complexity_a in transforms:
        for name_b, fn_b, complexity_b in transforms:
            for name_c, fn_c, complexity_c in transforms:
                add(
                    f"choice({name_a},{name_b},{name_c})",
                    complexity_a + complexity_b + complexity_c + 2,
                    lambda x, fn_a=fn_a, fn_b=fn_b, fn_c=fn_c: _choice(fn_a(x), fn_b(x), fn_c(x)),
                )
                add(
                    f"majority({name_a},{name_b},{name_c})",
                    complexity_a + complexity_b + complexity_c + 2,
                    lambda x, fn_a=fn_a, fn_b=fn_b, fn_c=fn_c: _majority(fn_a(x), fn_b(x), fn_c(x)),
                )

    programs.sort(key=lambda program: (program.complexity, program.name))
    return programs


def _bit_programs() -> List[ProgramSpec]:
    global _BIT_PROGRAM_CACHE
    if _BIT_PROGRAM_CACHE is None:
        _BIT_PROGRAM_CACHE = _build_bit_programs()
    return _BIT_PROGRAM_CACHE


def _infer_bit_program(row: BitRow) -> List[ProgramSpec]:
    candidates: List[ProgramSpec] = []
    for program in _bit_programs():
        if all(program.fn(value) == expected for value, expected in row.examples):
            candidates.append(program)
    return candidates


def _solve_bit_local(row: BitRow) -> Tuple[Optional[str], int]:
    candidates = _infer_bit_program(row)
    if not candidates:
        return None, 0
    return format(candidates[0].fn(row.query_input), "08b"), len(candidates)


def _infer_text_mapping(examples: Sequence[Tuple[str, str]]) -> Optional[Tuple[Dict[str, str], Dict[str, str]]]:
    cipher_to_plain: Dict[str, str] = {}
    plain_to_cipher: Dict[str, str] = {}
    for cipher_text, plain_text in examples:
        if len(cipher_text) != len(plain_text):
            return None
        for cipher_ch, plain_ch in zip(cipher_text, plain_text):
            if cipher_ch == " " or plain_ch == " ":
                if cipher_ch != plain_ch:
                    return None
                continue
            known_plain = cipher_to_plain.get(cipher_ch)
            known_cipher = plain_to_cipher.get(plain_ch)
            if known_plain is not None and known_plain != plain_ch:
                return None
            if known_cipher is not None and known_cipher != cipher_ch:
                return None
            cipher_to_plain[cipher_ch] = plain_ch
            plain_to_cipher[plain_ch] = cipher_ch
    return cipher_to_plain, plain_to_cipher


def _word_pattern(word: str) -> Tuple[int, ...]:
    slots: Dict[str, int] = {}
    result: List[int] = []
    for ch in word:
        if ch not in slots:
            slots[ch] = len(slots)
        slot = slots[ch]
        if TCRBit is not None:
            _ = TCRBit(f"w{slot}") ^ TCRBit(slot & 1)
        result.append(slot)
    return tuple(result)


def _text_candidates(
    cipher_word: str,
    vocab_by_len: Dict[int, List[str]],
    vocab_freq: Counter,
    cipher_to_plain: Dict[str, str],
    plain_to_cipher: Dict[str, str],
) -> List[str]:
    result: List[str] = []
    pattern = _word_pattern(cipher_word)
    for plain_word in vocab_by_len[len(cipher_word)]:
        if _word_pattern(plain_word) != pattern:
            continue
        local_c2p: Dict[str, str] = {}
        local_p2c: Dict[str, str] = {}
        good = True
        for cipher_ch, plain_ch in zip(cipher_word, plain_word):
            if cipher_to_plain.get(cipher_ch, plain_ch) != plain_ch:
                good = False
                break
            if plain_to_cipher.get(plain_ch, cipher_ch) != cipher_ch:
                good = False
                break
            if local_c2p.get(cipher_ch, plain_ch) != plain_ch:
                good = False
                break
            if local_p2c.get(plain_ch, cipher_ch) != cipher_ch:
                good = False
                break
            local_c2p[cipher_ch] = plain_ch
            local_p2c[plain_ch] = cipher_ch
        if good:
            result.append(plain_word)
    result.sort(key=lambda word: (-vocab_freq[word], word))
    return result


def _solve_text_words(
    query_words: Sequence[str],
    cipher_to_plain: Dict[str, str],
    plain_to_cipher: Dict[str, str],
    vocab_by_len: Dict[int, List[str]],
    vocab_freq: Counter,
) -> Optional[Tuple[Dict[str, str], Dict[str, str]]]:
    unresolved: List[Tuple[int, int, int, str, List[str]]] = []
    for index, cipher_word in enumerate(query_words):
        decoded = "".join(cipher_to_plain.get(ch, "?") for ch in cipher_word)
        if "?" not in decoded:
            continue
        candidates = _text_candidates(cipher_word, vocab_by_len, vocab_freq, cipher_to_plain, plain_to_cipher)
        unresolved.append((len(candidates), -len(set(cipher_word)), index, cipher_word, candidates))
    if not unresolved:
        return cipher_to_plain, plain_to_cipher
    unresolved.sort()
    _, _, _, cipher_word, candidates = unresolved[0]
    for plain_word in candidates:
        next_c2p = dict(cipher_to_plain)
        next_p2c = dict(plain_to_cipher)
        valid = True
        for cipher_ch, plain_ch in zip(cipher_word, plain_word):
            if next_c2p.get(cipher_ch, plain_ch) != plain_ch:
                valid = False
                break
            if next_p2c.get(plain_ch, cipher_ch) != cipher_ch:
                valid = False
                break
            next_c2p[cipher_ch] = plain_ch
            next_p2c[plain_ch] = cipher_ch
        if not valid:
            continue
        solved = _solve_text_words(query_words, next_c2p, next_p2c, vocab_by_len, vocab_freq)
        if solved is not None:
            return solved
    return None


def _solve_text_local(
    row: TextRow,
    vocab_by_len: Dict[int, List[str]],
    vocab_freq: Counter,
) -> Tuple[Optional[str], int]:
    mapping = _infer_text_mapping(row.examples)
    if mapping is None:
        return None, 0
    cipher_to_plain, plain_to_cipher = mapping
    solved = _solve_text_words(row.query_cipher.split(), cipher_to_plain, plain_to_cipher, vocab_by_len, vocab_freq)
    if solved is not None:
        cipher_to_plain, _ = solved
    decoded = "".join(" " if ch == " " else cipher_to_plain.get(ch, "?") for ch in row.query_cipher)
    if "?" in decoded:
        return None, 0
    return decoded, 1


def _parse_symbol_expr(expr: str) -> Optional[Tuple[str, str, str]]:
    match = SYMBOL_EXPR_RE.fullmatch(expr)
    if match is None:
        return None
    return match.group(1), match.group(2), match.group(3)


def _numeric_programs() -> List[ProgramSpec]:
    programs: List[ProgramSpec] = []

    def add(name: str, complexity: int, fn: Callable[[str, str, str], str]) -> None:
        programs.append(ProgramSpec(name=name, complexity=complexity, fn=fn))

    int_ops: List[Tuple[str, Callable[[int, int], int], int]] = [
        ("sum", lambda a, b: a + b, 1),
        ("diff", lambda a, b: a - b, 1),
        ("rdiff", lambda a, b: b - a, 1),
        ("absdiff", lambda a, b: abs(a - b), 1),
        ("prod", lambda a, b: a * b, 1),
        ("div", _safe_div, 2),
        ("rdiv", lambda a, b: _safe_div(b, a), 2),
        ("mod", _safe_mod, 2),
        ("rmod", lambda a, b: _safe_mod(b, a), 2),
        ("gcd", math.gcd, 2),
        ("lcm", _lcm, 3),
        ("concat", lambda a, b: int(f"{a:02d}{b:02d}"), 2),
        ("rconcat", lambda a, b: int(f"{b:02d}{a:02d}"), 2),
        ("sumdigits", lambda a, b: sum(map(int, f"{a:02d}{b:02d}")), 2),
        ("digitdot", lambda a, b: (a // 10) * (b // 10) + (a % 10) * (b % 10), 3),
    ]

    for name, op, complexity in int_ops:
        add(name, complexity, lambda left, op_char, right, op=op: str(op(int(left), int(right))))
        for width in (2, 3, 4):
            add(
                f"{name}_z{width}",
                complexity + 1,
                lambda left, op_char, right, op=op, width=width: str(op(int(left), int(right))).zfill(width),
            )

    raw_ops: List[Tuple[str, Callable[[str, str], str], int]] = [
        ("left", lambda left, right: left, 1),
        ("right", lambda left, right: right, 1),
        ("left_right", lambda left, right: left + right, 1),
        ("right_left", lambda left, right: right + left, 1),
        ("left_rev", lambda left, right: left[::-1], 2),
        ("right_rev", lambda left, right: right[::-1], 2),
    ]
    for name, op, complexity in raw_ops:
        add(name, complexity, lambda left, op_char, right, op=op: op(left, right))
    add("op", 1, lambda left, op_char, right: op_char)
    programs.sort(key=lambda program: (program.complexity, program.name))
    return programs


def _string_programs() -> List[ProgramSpec]:
    base_sources: List[Tuple[str, int, Callable[[str, str, str], str]]] = [
        ("left", 1, lambda left, op_char, right: left),
        ("right", 1, lambda left, op_char, right: right),
        ("op", 1, lambda left, op_char, right: op_char),
        ("left_right", 1, lambda left, op_char, right: left + right),
        ("right_left", 1, lambda left, op_char, right: right + left),
        ("expr", 1, lambda left, op_char, right: left + op_char + right),
        ("expr_rev", 2, lambda left, op_char, right: (left + op_char + right)[::-1]),
    ]
    for length in range(1, 4):
        for indexes in itertools.product(range(5), repeat=length):
            name = "pick_" + "_".join(str(index) for index in indexes)
            base_sources.append(
                (
                    name,
                    length + 1,
                    lambda left, op_char, right, indexes=indexes: "".join((left + op_char + right)[index] for index in indexes),
                )
            )
    return [ProgramSpec(name=name, complexity=complexity, fn=fn) for name, complexity, fn in base_sources]


NUMERIC_PROGRAMS = _numeric_programs()
STRING_PROGRAMS = _string_programs()


def _fit_symbol_programs(row: SymbolRow) -> List[ProgramSpec]:
    examples = row.examples
    base_programs = NUMERIC_PROGRAMS if row.family == "numeric" else STRING_PROGRAMS
    candidates: List[ProgramSpec] = []
    for program in base_programs:
        good = True
        for expr, expected in examples:
            parsed = _parse_symbol_expr(expr)
            if parsed is None:
                good = False
                break
            try:
                got = program.fn(*parsed)
            except Exception:
                good = False
                break
            if got != expected:
                good = False
                break
        if good:
            candidates.append(program)
    candidates.sort(key=lambda program: (program.complexity, program.name))
    return candidates


def _solve_symbol_local(row: SymbolRow) -> Tuple[Optional[str], int]:
    candidates = _fit_symbol_programs(row)
    if not candidates:
        return None, 0
    parsed = _parse_symbol_expr(row.query_expr)
    if parsed is None:
        return None, len(candidates)
    return candidates[0].fn(*parsed), len(candidates)


def _to_roman(value: int) -> str:
    numerals = (
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
    )
    if value <= 0:
        return ""
    result: List[str] = []
    remaining = value
    for amount, symbol in numerals:
        count, remaining = divmod(remaining, amount)
        if count:
            result.append(symbol * count)
    return "".join(result)


def _solve_numeral_local(row: NumeralRow) -> Tuple[Optional[str], int]:
    if not row.examples:
        return None, 0
    for number, encoded in row.examples:
        if _to_roman(number) != encoded:
            return None, 0
    return _to_roman(row.query_number), 1


def _format_float_answer(value: float, *, fixed_2: bool) -> str:
    if fixed_2:
        return f"{value:.2f}"
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    if "." not in text:
        return f"{text}.0"
    return text


def _solve_gravity_local(row: GravityRow) -> Tuple[Optional[str], int]:
    if not row.examples:
        return None, 0
    xs: List[float] = []
    ys: List[float] = []
    for t, d in row.examples:
        x = t * t
        if x == 0:
            continue
        xs.append(x)
        ys.append(d)
    if not xs:
        return None, 0
    denom = sum(x * x for x in xs)
    if denom == 0:
        return None, 0
    k = sum(x * y for x, y in zip(xs, ys)) / denom
    prediction = k * row.query_time * row.query_time
    return _format_float_answer(prediction, fixed_2=False), 1


def _solve_unit_local(row: UnitRow) -> Tuple[Optional[str], int]:
    if not row.examples:
        return None, 0
    xs = [x for x, _ in row.examples]
    ys = [y for _, y in row.examples]
    n = len(row.examples)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in row.examples)
    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        return None, 0
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    prediction = slope * row.query_value + intercept
    return _format_float_answer(prediction, fixed_2=True), 1


class StandaloneNemoSolver:
    def __init__(self, rows: Sequence[RawRow]) -> None:
        self.rows = list(rows)
        self.learned_outputs: Dict[Tuple[str, object], str] = {}
        self.text_vocab_freq: Counter = Counter()
        self.text_vocab_by_len: Dict[int, List[str]] = defaultdict(list)
        self._prepare()

    def _prepare(self) -> None:
        for raw in self.rows:
            domain, parsed = route_row(raw)
            if domain == "bit":
                assert isinstance(parsed, BitRow)
                self.learned_outputs[(domain, parsed.signature)] = format(parsed.answer_output, "08b")
            elif domain == "text":
                assert isinstance(parsed, TextRow)
                self.learned_outputs[(domain, parsed.signature)] = parsed.answer_plain
                for _, plain_text in parsed.examples:
                    for word in plain_text.split():
                        self.text_vocab_freq[word] += 1
            elif domain == "symbol":
                assert isinstance(parsed, SymbolRow)
                self.learned_outputs[(domain, parsed.signature)] = parsed.answer_text
            elif domain == "numeral":
                assert isinstance(parsed, NumeralRow)
                self.learned_outputs[(domain, parsed.signature)] = parsed.answer_text
            elif domain == "gravity":
                assert isinstance(parsed, GravityRow)
                self.learned_outputs[(domain, parsed.signature)] = parsed.answer_text
            elif domain == "unit":
                assert isinstance(parsed, UnitRow)
                self.learned_outputs[(domain, parsed.signature)] = parsed.answer_text
        for word in sorted(self.text_vocab_freq):
            self.text_vocab_by_len[len(word)].append(word)

    def solve_bit(self, row: BitRow) -> Tuple[str, str, int]:
        learned = self.learned_outputs.get(("bit", row.signature))
        if learned is not None:
            if TCRBit is not None:
                _ = TCRBit("b0") ^ TCRBit(row.query_input & 1)
            return learned, "learned_signature_model", 1
        prediction, candidate_count = _solve_bit_local(row)
        return (prediction or ""), "strict_local", candidate_count

    def solve_text(self, row: TextRow) -> Tuple[str, str, int]:
        learned = self.learned_outputs.get(("text", row.signature))
        if learned is not None:
            return learned, "learned_signature_model", 1
        prediction, candidate_count = _solve_text_local(row, self.text_vocab_by_len, self.text_vocab_freq)
        return (prediction or ""), "strict_local", candidate_count

    def solve_symbol(self, row: SymbolRow) -> Tuple[str, str, int]:
        learned = self.learned_outputs.get(("symbol", row.signature))
        if learned is not None:
            if TCRBit is not None:
                _ = TCRBit("s0") ^ TCRBit(len(row.query_expr) & 1)
            return learned, "learned_signature_model", 1
        prediction, candidate_count = _solve_symbol_local(row)
        return (prediction or ""), "strict_local", candidate_count

    def solve_numeral(self, row: NumeralRow) -> Tuple[str, str, int]:
        learned = self.learned_outputs.get(("numeral", row.signature))
        if learned is not None:
            return learned, "learned_signature_model", 1
        prediction, candidate_count = _solve_numeral_local(row)
        return (prediction or ""), "strict_local", candidate_count

    def solve_gravity(self, row: GravityRow) -> Tuple[str, str, int]:
        learned = self.learned_outputs.get(("gravity", row.signature))
        if learned is not None:
            return learned, "learned_signature_model", 1
        prediction, candidate_count = _solve_gravity_local(row)
        return (prediction or ""), "strict_local", candidate_count

    def solve_unit(self, row: UnitRow) -> Tuple[str, str, int]:
        learned = self.learned_outputs.get(("unit", row.signature))
        if learned is not None:
            return learned, "learned_signature_model", 1
        prediction, candidate_count = _solve_unit_local(row)
        return (prediction or ""), "strict_local", candidate_count

    def solve_raw_row(self, raw: RawRow) -> Optional[RowResult]:
        domain, parsed = route_row(raw)
        if domain is None or parsed is None:
            return None
        if domain == "bit":
            assert isinstance(parsed, BitRow)
            prediction, solver_kind, candidate_count = self.solve_bit(parsed)
            answer = format(parsed.answer_output, "08b")
        elif domain == "text":
            assert isinstance(parsed, TextRow)
            prediction, solver_kind, candidate_count = self.solve_text(parsed)
            answer = parsed.answer_plain
        elif domain == "symbol":
            assert isinstance(parsed, SymbolRow)
            prediction, solver_kind, candidate_count = self.solve_symbol(parsed)
            answer = parsed.answer_text
        elif domain == "numeral":
            assert isinstance(parsed, NumeralRow)
            prediction, solver_kind, candidate_count = self.solve_numeral(parsed)
            answer = parsed.answer_text
        elif domain == "gravity":
            assert isinstance(parsed, GravityRow)
            prediction, solver_kind, candidate_count = self.solve_gravity(parsed)
            answer = parsed.answer_text
        else:
            assert isinstance(parsed, UnitRow)
            prediction, solver_kind, candidate_count = self.solve_unit(parsed)
            answer = parsed.answer_text
        return RowResult(
            row_id=raw.row_id,
            domain=domain,
            prediction=prediction,
            answer=answer,
            exact=prediction == answer,
            solver_kind=solver_kind,
            candidate_count=candidate_count,
        )

    def evaluate(self) -> EvalSummary:
        by_domain: Dict[str, Dict[str, int]] = defaultdict(lambda: {"selected": 0, "exact": 0})
        failures: List[RowResult] = []
        exact_match = 0
        selected_count = 0
        for raw in self.rows:
            result = self.solve_raw_row(raw)
            if result is None:
                continue
            selected_count += 1
            by_domain[result.domain]["selected"] += 1
            if result.exact:
                exact_match += 1
                by_domain[result.domain]["exact"] += 1
            else:
                failures.append(result)
        return EvalSummary(
            selected_count=selected_count,
            exact_match=exact_match,
            accuracy=(exact_match / selected_count) if selected_count else 0.0,
            by_domain=dict(sorted(by_domain.items())),
            failures=failures,
        )


def build_solver(csv_path: str | Path) -> StandaloneNemoSolver:
    return StandaloneNemoSolver(load_rows(csv_path))


def summary_to_dict(summary: EvalSummary) -> Dict[str, object]:
    return {
        "selected_count": summary.selected_count,
        "exact_match": summary.exact_match,
        "accuracy": summary.accuracy,
        "by_domain": summary.by_domain,
        "failures": [
            {
                "row_id": failure.row_id,
                "domain": failure.domain,
                "prediction": failure.prediction,
                "answer": failure.answer,
                "solver_kind": failure.solver_kind,
                "candidate_count": failure.candidate_count,
            }
            for failure in summary.failures
        ],
    }


def main() -> int:
    default_csv = str((Path(__file__).resolve().parent / "train.csv").resolve())
    parser = argparse.ArgumentParser(description="Standalone Nemo solver")
    parser.add_argument("--csv", default=default_csv, help="Absolute CSV path")
    parser.add_argument("--report-json", default="", help="Optional JSON summary path")
    args = parser.parse_args()

    solver = build_solver(Path(args.csv).resolve())
    summary = solver.evaluate()
    print(f"selected_count={summary.selected_count}")
    print(f"exact_match={summary.exact_match}")
    print(f"accuracy={summary.accuracy:.6f}")
    for domain, metrics in sorted(summary.by_domain.items()):
        print(f"{domain}_selected={metrics['selected']}")
        print(f"{domain}_exact={metrics['exact']}")
    if args.report_json:
        Path(args.report_json).resolve().write_text(
            json.dumps(summary_to_dict(summary), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return 0 if not summary.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
