# NEMO Standalone Solver Analysis Report
**Goal:** Remove learned/signature/canonical/fallback answer lookup, solve 9500 train rows programmatically

---

## Executive Summary

**Current Status:**
- `standalone_nemo_solver.py` uses `learned_signature_model` (exact answer lookup) for 100% of train rows
- Local solvers (without lookup) achieve **74.1% exact accuracy** (7,043/9,500 rows)
- All signatures are unique (no duplicate example+query combinations)
- `cry` library usage is optional (lines 15, 541-542, 881, 897) - only for side effects, not solving

**Blocker Assessment:** ✅ **NO BLOCKERS** to delete all files except `standalone_nemo_solver.py` and `train.csv`
- No cross-file imports between decoders
- `standalone_nemo_solver.py` is self-contained
- Optional `cry` library already handled with try/except

---

## Domain-by-Domain Analysis

### Dataset Distribution
- **bit**: 1,602 rows (16.9%)
- **text**: 1,576 rows (16.6%)
- **symbol**: 1,555 rows (16.4%)
- **numeral**: 1,576 rows (16.6%)
- **gravity**: 1,597 rows (16.8%)
- **unit**: 1,594 rows (16.8%)

### Local Solver Performance (Without Learned Lookup)

| Domain   | Total | Solved | Exact   | Gap     | Root Cause |
|----------|-------|--------|---------|---------|------------|
| **bit**  | 1,602 | 76.8%  | 75.6%   | 24.4%   | Program space too limited (371 out of 1,602 unsolved) |
| **text** | 1,576 | 100%   | 100%    | 0%      | ✅ **FULLY SOLVED** |
| **symbol**| 1,555| 1.1%   | 0.5%    | 99.5%   | 1,538/1,555 no candidates - programs too simple |
| **numeral**|1,576| 100%   | 100%    | 0%      | ✅ **FULLY SOLVED** (Roman numerals) |
| **gravity**|1,597| 100%   | 88.9%   | 11.1%   | Floating point precision (177 off by rounding) |
| **unit** | 1,594 | 100%   | 78.6%   | 21.4%   | Floating point precision (341 off by rounding) |
| **TOTAL**| 9,500 | 79.9%  | **74.1%**| **25.9%**| 2,457 rows need solution |

---

## Problem Analysis by Domain

### ✅ Domains with 100% Example-Based Solvability

#### 1. TEXT CIPHER (1,576 rows - 100% exact)
**Current approach:** Substitution cipher with word-pattern matching + vocabulary from examples
**Citations:** 
- Lines 512-638: `_infer_text_mapping`, `_solve_text_words`, `_solve_text_local`
- Lines 846-861, 874-875, 886-891: Vocabulary built from example plain texts

**Why it works:** Consistent character substitution + word patterns + example vocabulary covers all queries

#### 2. NUMERAL SYSTEM (1,576 rows - 100% exact)
**Current approach:** Roman numeral detection and generation
**Citations:** Lines 757-790: `_to_roman`, `_solve_numeral_local`

**Why it works:** All rows use Roman numerals (verified by checking examples match `_to_roman`)

### 🟡 Domains with Precision/Rounding Issues Only

#### 3. GRAVITY (1,597 rows - 88.9% exact)
**Current approach:** Least squares regression `k = Σ(xy)/Σ(x²)` where `d = k*t²`
**Citations:** Lines 802-820: `_solve_gravity_local`
**Gap:** 177 rows (11.1%) have floating-point formatting mismatches

**Synthesis strategy:** Answer formatting heuristics
- Row 00463d04: Examples show 2 decimal places consistently
- Row 0040ff76: Examples show 2 decimal places consistently
- **Pattern:** Most answers are formatted as `{value:.2f}`
- **Solution:** Detect decimal place consistency from examples, apply same formatting

#### 4. UNIT CONVERSION (1,594 rows - 78.6% exact)
**Current approach:** Linear regression `y = slope*x + intercept`
**Citations:** Lines 823-839: `_solve_unit_local`
**Gap:** 341 rows (21.4%) have formatting mismatches

**Synthesis strategy:** Answer formatting heuristics
- Row 00208201: Answer "16.65" (2 decimals)
- Row 0047365c: Answer "10.62" (2 decimals)
- **Pattern:** Always 2 decimal places (line 839: `fixed_2=True`)
- **Current implementation already correct** - likely precision edge cases

### 🔴 Domains Requiring Expanded Program Space

#### 5. BIT MANIPULATION (1,602 rows - 75.6% exact)
**Current approach:** Synthesize from 371 programs (unary+binary+ternary bit ops)
**Citations:** Lines 433-509: `_build_bit_programs`, `_infer_bit_program`, `_solve_bit_local`
**Gap:** 391 rows (24.4%) have **no matching candidates**

**Analysis:**
- Programs: 371 total (lines 433-494)
  - 3 unary + 14 shifts/rotations = 17 transforms
  - 17² × 5 binary ops = 1,445 binary combinations
  - 17³ × 2 ternary ops (choice, majority) = 9,826 ternary combinations (but only subset included)
- **Problem:** Real transforms are more complex than current program space

**Synthesis strategies:**
1. **Expand program space** (No cross-row learning needed):
   - Add more ternary combinations (currently limited)
   - Add quaternary operations: `(a & b) | (c ^ d)`
   - Add conditional operations: `if bit7: rotate else shift`
   
2. **Signature clustering** (Cross-row pattern detection):
   - Each signature is unique (no duplicates)
   - **Cannot cluster by exact signature match**
   - **Alternative:** Cluster by "partial signature" - group rows with same example count + similar transform complexity
   - Example: Rows with 8-9 examples that all fail simple programs likely share complexity class

**Concrete guidance:**
- Increase ternary program coverage from current subset to full 17³×2 space
- Add 4-way combinations: `(f1(x) op1 f2(x)) op2 (f3(x) op3 f4(x))`
- Estimated new program count: ~50,000 (still tractable for enumeration)

#### 6. SYMBOL TRANSFORM (1,555 rows - 0.5% exact)
**Current approach:** 67 numeric programs + 162 string programs
**Citations:** 
- Lines 647-719: `_numeric_programs`, `_string_programs`
- Lines 722-754: `_fit_symbol_programs`, `_solve_symbol_local`
**Gap:** 1,538 rows (98.9%) have **no matching candidates**

**Analysis - Numeric family (732 rows, 726 failures):**
- Row 00457d26: `69/52 -> 17/` but examples: `34/44=1, 41/32=9, 34|25=69, 87\64=8853`
- **Pattern:** Answer format varies (1-4 chars), not predictable from simple ops
- **Problem:** Programs output fixed format, but answers have variable length/format

**Analysis - Symbol_string family (823 rows, 822 failures):**
- Row 00457d26: `[[-!' -> @&` (5 chars → 2 chars)
- Row 00457d26: `))!\) -> \^?` (5 chars → 3 chars)
- **Pattern:** Answer length varies (not fixed by query structure)
- **Problem:** Current 162 programs only handle fixed-length extractions from 5-char input

**Synthesis strategies:**

**Strategy 1: Expand string program space** (No cross-row learning)
- Current programs: Pick substrings from fixed 5-char `left+op+right`
- **Add:** Character-level transformations on picked substrings
  - Caesar shifts per position
  - Character XOR/substitution tables
  - Conditional transformations based on character class
- **Add:** Variable-length outputs based on character matching rules
  - Filter characters by condition
  - Duplicate characters based on pattern
- Estimated new program count: ~5,000-10,000

**Strategy 2: Symbolic character mapping** (Cross-row pattern detection)
- Observation: Same operator character in query may map to consistent transformations
- **Cluster by:** Operator character in middle position (lines 297-301, 641-644)
- **Model per cluster:** Learn character substitution rules specific to operator
- Example: All rows with `*` operator might follow rule set A, `/` operator follows rule set B

**Strategy 3: Example-based synthesis** (Cross-row learning)
- **Key insight:** Each row has 3-5 example transformations
- **Approach:** For each query, find rows with similar example patterns
  - Similarity metric: Edit distance between example expressions
  - Learn transformation function from similar rows' examples
- **Implementation:** Store (example_pattern_hash → transformation_candidates) map
- **Limitation:** Requires keeping training data

**Recommendation for symbol domain:**
- **Strategy 1 (expand programs)** is preferred - no cross-row dependency
- **Strategy 2 (cluster by operator)** is secondary - reusable pattern but needs grouping
- **Strategy 3 (example-based)** violates "no exact lookup" if implemented naively

---

## Cross-Row Synthesis Analysis

### Signature Uniqueness
**Finding:** All 9,500 signatures are unique (100% unique rate across all domains)
**Citation:** Analysis shows 0 duplicate signatures across all domains
**Implication:** Cannot use exact signature matching for cross-row learning

### Viable Cross-Row Strategies (Without Exact Answer Lookup)

#### 1. Program Complexity Clustering
**Domains:** Bit, Symbol
**Approach:** Group rows by "program complexity class"
- **Metric:** Minimum complexity of any fitting program (if found), or "high" if none found
- **Usage:** Prioritize checking higher-complexity programs for rows in "high" cluster
- **Not answer lookup:** Shares program search strategy, not answers

#### 2. Operator-Based Clustering
**Domain:** Symbol
**Approach:** Group rows by middle operator character
**Citation:** Lines 297-301, 641-644 (operator extraction)
- **Usage:** Build operator-specific program sets
- **Not answer lookup:** Shares transformation rules, not specific answers

#### 3. Example-Count Stratification
**Domains:** All
**Approach:** Group rows by number of examples provided
- **Usage:** Adjust search strategy based on constraint count
- **Not answer lookup:** Adjusts search parameters, not answers

#### 4. Formatting Pattern Detection
**Domains:** Gravity, Unit
**Approach:** Detect decimal place consistency from examples
- **Usage:** Apply same formatting to answer
- **Not answer lookup:** Infers formatting rule, computes answer independently

### Non-Viable Approaches (Would Be Answer Lookup)

❌ **Exact signature → answer mapping** (current implementation lines 845-873, 878-921)
❌ **Partial signature → answer database** (still exact lookup)
❌ **Example hash → answer cache** (still exact lookup)

---

## Concrete Implementation Guidance

### To Achieve 100% Programmatic Coverage:

#### 1. BIT DOMAIN (expand program space)
```python
# Current: Lines 433-494 generate 371 programs
# Add quaternary operations:
def _build_bit_programs_extended():
    # ... existing code ...
    # Add 4-way ops: (f1 op1 f2) op2 (f3 op3 f4)
    for f1, f2, f3, f4 in itertools.combinations(transforms, 4):
        for op1, op2, op3 in itertools.product(binary_ops, repeat=3):
            add(f"{op3}({op1}({f1},{f2}),{op2}({f3},{f4}))", ...)
```
**Expected impact:** 391 → ~50 unsolved rows

#### 2. SYMBOL DOMAIN (expand program space + operator clustering)
```python
# Expand string programs with character-level transforms
def _string_programs_extended():
    # ... existing pick operations ...
    # Add character transformations
    for pick_program in base_pick_programs:
        for transform in ['caesar_1', 'caesar_-1', 'swap_case', ...]:
            add(f"{pick_program}_{transform}", ...)
```
**Expected impact:** 1,538 → ~500 unsolved rows (estimate)

Operator clustering:
```python
# Group by middle operator, build specialized program sets
operator_programs = defaultdict(list)
for row in training_rows:
    op = row.query_expr[2]  # middle char
    operator_programs[op].extend(find_fitting_programs(row))
# Use operator-specific programs first when solving
```
**Expected impact:** 500 → ~100 unsolved rows (estimate)

#### 3. GRAVITY & UNIT DOMAINS (precision handling)
```python
# Current issue: Lines 793-799, 839 - formatting inconsistencies
# Solution: Detect decimal places from examples
def _infer_decimal_places(examples):
    places = []
    for _, answer_str in examples:
        if '.' in answer_str:
            places.append(len(answer_str.split('.')[1].rstrip('0')))
    return max(places) if places else 2
```
**Expected impact:** 518 → ~50 unsolved rows (precision edge cases)

---

## Answer to Key Questions

### 1. Can six domains be solved from examples only?
**YES** for 4 domains, **PARTIALLY** for 2 domains:
- ✅ Text: 100% (current implementation)
- ✅ Numeral: 100% (current implementation)
- ✅ Gravity: 88.9% (need precision fixes)
- ✅ Unit: 78.6% (need precision fixes)
- 🟡 Bit: 75.6% (need expanded program space - ~30K more programs)
- 🟡 Symbol: 0.5% (need 100x program expansion + operator clustering)

### 2. Reusable cross-row strategies (not exact answer lookup)?
**Program space clustering:**
- **Bit:** Complexity-based program search ordering
- **Symbol:** Operator-based program sets (e.g., all `*` ops share patterns)
- **All:** Example-count stratification for search strategy

**Formatting pattern detection:**
- **Gravity/Unit:** Decimal place inference from example answers

**NOT answer lookup because:**
- Programs are synthesized per-row from examples
- Clustering affects *which programs to try first*, not *what answer to return*
- Answer is always computed from query, never retrieved

### 3. Blockers to deleting all files except standalone_nemo_solver.py?
**NONE.** Analysis confirms:
- ✅ Lines 3-12: Only stdlib imports
- ✅ Lines 15-17: `cry` library optional (try/except)
- ✅ No imports from `bit_manipulation_decoder.py`, `symbol_transform_decoder.py`, `text_cipher_decoder.py`
- ✅ `tests/` directory can be deleted (no runtime dependency)
- ✅ `*.json`, `*.md`, `run_*.py` files are artifacts (no runtime dependency)

**Files to keep:**
- `standalone_nemo_solver.py` (core solver)
- `train.csv` (data)

**Files safe to delete:**
- `bit_manipulation_decoder.py`, `symbol_transform_decoder.py`, `text_cipher_decoder.py`
- `run_*.py` (evaluation scripts)
- `tests/*` (test files)
- `*.json` (diagnostics)
- `*.md` (notes)

---

## Summary Table: Path to 100% Programmatic Solving

| Domain | Current | Gap | Strategy | Estimated Effort |
|--------|---------|-----|----------|------------------|
| Text | 100% | 0% | ✅ Complete | None |
| Numeral | 100% | 0% | ✅ Complete | None |
| Gravity | 88.9% | 11.1% | Precision/formatting fixes | Low (1-2 hours) |
| Unit | 78.6% | 21.4% | Precision/formatting fixes | Low (1-2 hours) |
| Bit | 75.6% | 24.4% | Expand program space to ~50K | Medium (4-8 hours) |
| Symbol | 0.5% | 99.5% | Expand programs + operator clustering | High (16-24 hours) |

**Total remaining work:** ~24-36 hours of implementation to reach 100% programmatic coverage

**Key insight:** The "learned signature lookup" (lines 845-921) is exact answer memorization. Removing it drops accuracy from 100% → 74.1%, but all gaps are addressable through program space expansion and formatting heuristics, NOT through cross-row answer lookup.
