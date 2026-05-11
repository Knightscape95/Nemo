# Text Cipher Decoder Notes

## Scope
- Dataset: `<repo-root>/train.csv` (default resolves to an absolute path at runtime)
- Target rows: prompt contains `secret encryption rules are used on text` and has `Now, decrypt the following text:`.

## Generator reconstruction assumptions
- Cipher is row-local monoalphabetic substitution at character level.
- Spaces are preserved.
- Mapping is bijective for observed letters (`cipher -> plain` and `plain -> cipher`).

## Deterministic decoding strategy
1. Parse multiline prompt safely with CSV reader.
2. Extract example pairs (`cipher -> plain`) and query cipher text.
3. Infer row mapping from examples and validate consistency.
4. Decode query directly.
5. If unresolved letters remain, run deterministic constrained fallback:
   - build candidate plaintext words from example plaintext vocabulary,
   - constrain by known partial mapping and bijection,
   - solve unresolved query words with deterministic backtracking order.

## Validation outputs
- Exact-match accuracy on all selected rows.
- Per-row diagnostics include unresolved counts and fallback usage.
- Failure buckets include inconsistent mapping, unresolved chars, and wrong prediction.

## Usage
```bash
python <repo-root>/run_text_cipher_eval.py --csv <repo-root>/train.csv
```

JSON diagnostics report is written to:
- `<repo-root>/text_cipher_error_analysis.json` (absolute path resolved by the script)
