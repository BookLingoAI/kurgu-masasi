"""Turkish text helpers: normalization, dictionary corrections, hallucination filter, token similarity."""
from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s']", re.UNICODE)


def tr_lower(s: str) -> str:
    return s.replace("I", "ı").replace("İ", "i").lower()


def norm_word(w: str) -> str:
    w = tr_lower(unicodedata.normalize("NFC", w))
    w = _PUNCT.sub("", w)
    return w.replace("'", "").strip()


def apply_dictionary(text: str, replacements: dict[str, str]) -> str:
    out = text
    for wrong, right in sorted(replacements.items(), key=lambda kv: -len(kv[0])):
        if wrong.startswith("_"):
            continue
        pat = re.compile(r"(?<!\w)" + re.escape(wrong) + r"(?!\w)", re.IGNORECASE | re.UNICODE)
        out = pat.sub(right, out)
    return out


def correct_words(words: list[dict], replacements: dict[str, str]) -> list[dict]:
    """Apply dictionary corrections on word lists (handles multi-word keys by merging spelling into the first word)."""
    if not replacements:
        return words
    keys = [(k, v) for k, v in replacements.items() if not k.startswith("_")]
    keys.sort(key=lambda kv: -len(kv[0].split()))
    i = 0
    out = []
    while i < len(words):
        matched = False
        for wrong, right in keys:
            parts = [norm_word(p) for p in wrong.split()]
            n = len(parts)
            if i + n <= len(words) and [norm_word(w["word"]) for w in words[i:i + n]] == parts:
                trail = re.search(r"[^\w']+$", words[i + n - 1]["word"].strip())
                rparts = right.split()
                if len(rparts) == n:
                    for j in range(n):
                        nw = dict(words[i + j])
                        nw["word"] = rparts[j] + (trail.group(0) if (j == n - 1 and trail) else "")
                        out.append(nw)
                else:
                    nw = dict(words[i])
                    nw["word"] = right + (trail.group(0) if trail else "")
                    nw["end"] = words[i + n - 1]["end"]
                    out.append(nw)
                i += n
                matched = True
                break
        if not matched:
            out.append(words[i])
            i += 1
    return out


def is_hallucination(text: str, patterns: list[str]) -> bool:
    t = tr_lower(text.strip())
    return any(re.search(p, t) for p in patterns)


def token_sim(a: str, b: str, min_prefix: int = 3, frac: float = 0.6) -> float:
    """1.0 equal, 0.8 same stem (suffix-corrected), else 0."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    if n >= max(min_prefix, int(frac * min(len(a), len(b)))):
        return 0.8
    return 0.0


def seq_sim(a: list[str], b: list[str], **kw) -> float:
    """Average positional token similarity of a against the first len(a) tokens of b."""
    if not a or len(b) < 1:
        return 0.0
    n = min(len(a), len(b))
    return sum(token_sim(a[i], b[i], **kw) for i in range(n)) / len(a)
