"""Cut rules (section 6). Produces removal intervals with reasons, then the kept segments.

Inputs are in one continuous source timeline (camera audio). Every removal carries a `rule` and human-readable
`reason`; the planner writes them to decisions.log.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Callable

from ..analysis.audio import merge_regions
from ..analysis.text import norm_word, seq_sim, token_sim

INTERJECTIONS = {"evet", "yani", "tamam", "şey", "neyse", "pardon", "hayır", "yok", "dur", "bir", "saniye"}
# "bir saniye" is handled as two interjection tokens


@dataclass
class Removal:
    start: float
    end: float
    rule: str
    reason: str
    words: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def ends_sentence(word: str) -> bool:
    """A completed sentence ('.', '!', '?') - an ellipsis ('...', '…') means the speaker trailed off."""
    w = word.rstrip()
    return w.endswith((".", "!", "?")) and not w.endswith(("...", "…"))


def _phrases(words: list[dict], gap: float) -> list[list[int]]:
    out: list[list[int]] = []
    for i, w in enumerate(words):
        if out and w["start"] - words[out[-1][-1]]["end"] < gap and not words[out[-1][-1]]["word"].rstrip().endswith((".", "?", "!")):
            out[-1].append(i)
        else:
            out.append([i])
    return out


# hesitation sounds, including elongated spellings Whisper produces ("eeeh", "ööööö…", "hıııı", "mmmh")
FILLER_RE = re.compile(r"^(ı+h*m*|ıh+m*|ee+h*m*|eh+m*|öö+h*m*|öh+m*|hm+|mm+h*|hıı+)$")


def is_filler(word: str, fillers: set[str]) -> bool:
    n = norm_word(word)
    if not n:
        return False
    collapsed = re.sub(r"(.)\1{2,}", r"\1\1\1", n)
    return n in fillers or collapsed in fillers or bool(FILLER_RE.match(n))


def find_fillers(words: list[dict], fillers: set[str]) -> tuple[list[Removal], set[int]]:
    rem, idx = [], set()
    for i, w in enumerate(words):
        if is_filler(w["word"], fillers):
            shown = w["word"] if len(w["word"]) <= 12 else w["word"][:10] + "…"
            rem.append(Removal(w["start"], w["end"], "filler", f"Dolgu sesi: '{shown}'", [shown]))
            idx.add(i)
    return rem, idx


# Whisper's Turkish "subtitle credits" it hears in music or noise (learned from YouTube subtitles)
HALLUCINATION = {"abone", "ol", "olun", "olmayı", "unutmayın", "altyazı", "mk", "m", "k", "izlediğiniz",
                 "beğenmeyi", "atmayı", "kanalıma", "kanala"}


def find_hallucinations(words: list[dict], removed: set[int]) -> tuple[list[Removal], set[int]]:
    """Runs of 1-6 words made only of those phrases, heard with low confidence: invented text, not speech (the
    speaker's real "Abone olun" was heard with 0.7-0.9 confidence, the invented one over the music with ~0.5)."""
    rem, idx = [], set()
    i = 0
    n = len(words)
    while i < n:
        if i in removed or norm_word(words[i]["word"]) not in HALLUCINATION:
            i += 1
            continue
        j = i
        while j + 1 < n and j + 1 - i < 6 and norm_word(words[j + 1]["word"]) in HALLUCINATION \
                and words[j + 1]["start"] - words[j]["end"] < 0.8:
            j += 1
        run = words[i:j + 1]
        probs = [w.get("prob") for w in run if w.get("prob") is not None]
        before = words[i - 1]["end"] if i > 0 else -9.0
        after = words[j + 1]["start"] if j + 1 < n else 1e9
        isolated = run[0]["start"] - before >= 0.6 and after - run[-1]["end"] >= 0.6
        mean = sum(probs) / len(probs) if probs else 1.0
        lone = isolated or run[0]["start"] - before >= 1.5
        if probs and (mean < 0.6 or (min(probs) < 0.58 and len(run) <= 3 and lone)) and \
                any(norm_word(w["word"]) in ("abone", "altyazı", "izlediğiniz") for w in run):
            shown = " ".join(w["word"] for w in run)[:30]
            rem.append(Removal(run[0]["start"], run[-1]["end"], "filler", f"Uydurma yazı (müzikte duyuldu): '{shown}'", [shown]))
            idx.update(range(i, j + 1))
        i = j + 1
    return rem, idx


def _coverage(a: float, b: float, active: list[tuple[float, float]]) -> float:
    if b <= a:
        return 0.0
    return sum(max(0.0, min(b, d) - max(a, c)) for c, d in active if d > a and c < b) / (b - a)


def find_nonspeech(active: list[tuple[float, float]], words: list[dict], max_len: float,
                   orphan_reach: float = 0.7) -> list[Removal]:
    """Energy islands that contain no word: cough, sneeze, laugh, breath, desk noise.

    Whisper word times can drift by ~0.5 s. A word whose span is mostly silent is 'orphaned': its sound is
    probably the neighbouring island, so islands within `orphan_reach` of an orphan word are not cut."""
    orphans = [w for w in words if _coverage(w["start"], w["end"], active) < 0.5]
    rem = []
    j = 0
    for a, b in active:
        covered = 0.0
        while j < len(words) and words[j]["end"] < a:
            j += 1
        k = j
        while k < len(words) and words[k]["start"] < b:
            covered += max(0.0, min(b, words[k]["end"]) - max(a, words[k]["start"]))
            k += 1
        near_orphan = any(w["start"] - orphan_reach < b and w["end"] + orphan_reach > a for w in orphans)
        if covered < 0.2 * (b - a) and (b - a) <= max_len and not near_orphan:
            rem.append(Removal(a, b, "nonspeech", f"Kelimesiz ses ({b - a:.2f} sn): öksürük/nefes/gülme/gürültü"))
    return rem


def _tokens(words, idxs):
    return [norm_word(words[i]["word"]) for i in idxs]


def find_repetitions(words: list[dict], cfg: dict, removed: set[int],
                     paraphrase_sim: Callable[[str, str], float] | None = None) -> tuple[list[Removal], set[int]]:
    """Keep-the-last-take repetition detection. Returns removals and removed word indices."""
    kw = {"min_prefix": cfg["stem_min_prefix"], "frac": cfg["stem_prefix_fraction"]}
    alive = [i for i in range(len(words)) if i not in removed]
    W = [words[i] for i in alive]
    toks = [norm_word(w["word"]) for w in W]
    dead: set[int] = set()
    rem: list[Removal] = []
    notes: list[Removal] = []

    def kill(a: int, b: int, rule: str, reason: str):
        """Remove alive-positions [a, b)."""
        if b <= a:
            return
        for p in range(a, b):
            dead.add(alive[p])
        rem.append(Removal(W[a]["start"], W[b]["start"] if b < len(W) else W[b - 1]["end"], rule, reason,
                           [W[p]["word"] for p in range(a, b)]))

    # (1) immediate n-gram repeats inside the flow, optionally separated by an interjection ("X Y evet X Y Z")
    def middle_ok(mid: list[str]) -> bool:
        if not mid:
            return True
        if mid == ["bir", "saniye"]:
            return True
        return len(mid) == 1 and mid[0] in INTERJECTIONS - {"bir", "saniye"}

    i = 0
    while i < len(W):
        hit = False
        for n in (4, 3, 2, 1):
            for gap_tokens in (0, 1, 2):
                j = i + n + gap_tokens
                if j + n > len(W) or not middle_ok(toks[i + n:j]):
                    continue
                if W[i + n]["start"] - W[i + n - 1]["end"] > cfg["inflow_max_gap_s"] and not gap_tokens:
                    continue  # copies separated by a real pause: handled by the restart rule (with containment check)
                if gap_tokens and W[j]["start"] - W[i + n - 1]["end"] > cfg["inflow_max_gap_s"] + 1.0:
                    continue
                a, b = toks[i:i + n], toks[j:j + n]
                sims = [token_sim(x, y, **kw) for x, y in zip(a, b)]
                if sims[0] == 0.0 or sims[-1] == 0.0 or sum(sims) / n < cfg["restart_similarity"]:
                    continue
                if any(ends_sentence(W[q]["word"]) for q in range(i, j)):
                    # first copy holds a finished sentence: not an in-flow retake (the restart rule decides)
                    if n >= 2:
                        notes.append(Removal(W[i]["start"], W[j + n - 1]["end"], "complete_kept",
                                             "Tamamlanmış cümle tekrar gibi görünüyor, tutuldu (kontrol et): '"
                                             + " ".join(W[q]["word"] for q in range(i, j + n)) + "'"))
                    continue
                if n == 1 and gap_tokens == 0:
                    gapt = W[j]["start"] - W[i]["end"]
                    if sims[0] == 1.0 and len(a[0]) >= 3 and gapt <= 0.35:
                        continue  # ikileme: "teker teker", "yavaş yavaş", "çok çok"
                    if sims[0] == 0.8 and b[0][len(a[0]):] in ("a", "e", "ya", "ye"):
                        continue  # ikileme with dative: "yan yana", "baş başa", "insan insana"
                if (n >= 2 and gap_tokens == 0 and all(x == 1.0 for x in sims)
                        and W[i + n - 1]["word"].rstrip().endswith((".", "!", "?"))
                        and W[j + n - 1]["word"].rstrip().endswith((".", "!", "?"))):
                    # a whole sentence said twice, both times completed: emphasis, not a retake
                    notes.append(Removal(W[i]["start"], W[j + n - 1]["end"], "emphasis_kept",
                                         f"Tam cümle iki kez söylenmiş, vurgu sayıldı ve tutuldu (kontrol et): "
                                         f"'{' '.join(W[p]['word'] for p in range(i, i + n))}'"))
                    i = j + n
                    hit = True
                    break
                kind = "suffix" if any(x == 0.8 for x in sims) else ("interjection" if gap_tokens else "repeat")
                why = {"suffix": "Eki düzeltilerek tekrar edilen kelime",
                       "interjection": "Araya söz girip yeniden başlanan ifade",
                       "repeat": "Hemen tekrar edilen ifade"}[kind]
                kill(i, j, "repetition", f"{why}: '{' '.join(W[p]['word'] for p in range(i, j))}' → son deneme tutuldu")
                i = j
                hit = True
                break
            if hit:
                break
        if not hit:
            i += 1

    # (2) sentence restarts across a pause: phrase A then phrase B that starts like A and contains A
    alive2 = [p for p in range(len(W)) if alive[p] not in dead]
    sub = [W[p] for p in alive2]
    phr = _phrases(sub, gap=0.25)
    for k in range(len(phr) - 1):
        A = phr[k]
        for step in (1, 2):
            if k + step >= len(phr):
                break
            B = phr[k + step]
            if step == 2:
                middle = phr[k + 1]
                mt = [norm_word(sub[q]["word"]) for q in middle]
                if len(middle) > 2 or not all(t in INTERJECTIONS for t in mt):
                    break
            if sub[B[0]]["start"] - sub[A[-1]]["end"] > cfg["repeat_window_s"]:
                break
            ta = [norm_word(sub[q]["word"]) for q in A]
            tb = [norm_word(sub[q]["word"]) for q in B]
            ms = cfg["restart_min_shared_words"]
            if len(ta) < ms and not (len(ta) >= 1 and len(tb) > len(ta)):
                continue
            head = sum(token_sim(x, y, **kw) for x, y in zip(ta[:ms], tb[:ms])) / max(1, min(ms, len(ta)))
            if head < 0.9:
                continue
            if len(tb) < len(ta):
                continue  # B shorter than A: not a restart, A is the complete take
            if seq_sim(ta, tb, **kw) < cfg["restart_similarity"]:
                continue  # lists / parallel phrases: "hangi model gelmiş, hangi model gitmiş"
            complete = ends_sentence(sub[A[-1]]["word"])
            if complete and ta != tb and ta != tb[:len(ta)]:
                continue  # a finished sentence is only a retake if the next one repeats it word for word
            if ta == tb and len(ta) >= 2 and complete:
                notes.append(Removal(sub[A[0]]["start"], sub[B[-1]]["end"], "emphasis_kept",
                                     "Tam cümle iki kez söylenmiş, vurgu sayıldı ve tutuldu (kontrol et): '"
                                     + " ".join(sub[q]["word"] for q in A) + "'"))
                break
            first = alive2[A[0]]
            last = alive2[B[0]]
            for p in range(first, last):
                dead.add(alive[p])
            mid_txt = " ".join(W[p]["word"] for p in range(first, last))
            rem.append(Removal(W[first]["start"], W[last]["start"], "restart",
                               f"Aynı açılışla yeniden başlanan cümle: '{mid_txt}' kesildi, son deneme tutuldu",
                               [W[p]["word"] for p in range(first, last)]))
            break

    # (3) paraphrase: consecutive sentences saying the same thing (needs embeddings)
    if paraphrase_sim is not None:
        alive3 = [p for p in range(len(W)) if alive[p] not in dead]
        sub3 = [W[p] for p in alive3]
        sents = _phrases(sub3, gap=0.6)
        for k in range(len(sents) - 1):
            A, B = sents[k], sents[k + 1]
            if len(A) < 4 or len(B) < 4:
                continue
            if sub3[B[0]]["start"] - sub3[A[-1]]["end"] > cfg["paraphrase_window_s"]:
                continue
            ta = " ".join(sub3[q]["word"] for q in A)
            tb = " ".join(sub3[q]["word"] for q in B)
            s = paraphrase_sim(ta, tb)
            if cfg["paraphrase_similarity"] <= s < cfg.get("paraphrase_cut_similarity", 0.98):
                # embeddings over-score sentences that share words but move the thought forward
                notes.append(Removal(sub3[A[0]]["start"], sub3[B[-1]]["end"], "paraphrase_review",
                                     f"Başka sözlerle aynı cümle olabilir (benzerlik {s:.2f}), kesilmedi - kontrol et: '{ta}'"))
                continue
            if s >= cfg.get("paraphrase_cut_similarity", 0.98):
                first, last = alive3[A[0]], alive3[B[0]]
                for p in range(first, last):
                    dead.add(alive[p])
                rem.append(Removal(W[first]["start"], W[last]["start"], "paraphrase",
                                   f"Başka sözlerle aynı cümle (benzerlik {s:.2f}): '{ta}' kesildi", [ta]))
    find_repetitions.notes = notes  # kept-but-flagged decisions for the log
    return rem, dead


def _outside(rems: list[Removal], protected: list[tuple[float, float]]) -> list[Removal]:
    """Silence removals with the protected spans (video playing on screen, food being shown) taken out of them."""
    if not protected:
        return rems
    out = []
    for r in rems:
        pieces = [(r.start, r.end)]
        for pa, pb in protected:
            nxt = []
            for a, b in pieces:
                if pb <= a or pa >= b:
                    nxt.append((a, b))
                    continue
                if pa - a >= 0.25:
                    nxt.append((a, pa))
                if b - pb >= 0.25:
                    nxt.append((pb, b))
            pieces = nxt
        for a, b in pieces:
            out.append(Removal(a, b, r.rule, r.reason, list(getattr(r, "words", []) or [])) if (a, b) != (r.start, r.end) else r)
    return out


def find_long_waits(words: list[dict], removed: set[int], total: float, min_len: float,
                    protected: list[tuple[float, float]]) -> list[Removal]:
    kept = [w for i, w in enumerate(words) if i not in removed]
    edges = [0.0] + [x for w in kept for x in (w["start"], w["end"])] + [total]
    rem = []
    for k in range(0, len(edges) - 1, 2):
        a, b = edges[k], edges[k + 1]
        if b - a < min_len:
            continue
        prot = sum(max(0.0, min(b, pb) - max(a, pa)) for pa, pb in protected)
        if prot > 0.5 * (b - a):
            continue
        rem.append(Removal(a, b, "long_wait", f"Hiç kelime olmayan {b - a:.1f} sn bekleme"))
    return rem


def find_pauses(active: list[tuple[float, float]], total: float, min_len: float, keep: float) -> list[Removal]:
    rem = []
    t = 0.0
    for a, b in active + [(total, total)]:
        if a - t >= min_len:
            rem.append(Removal(t + keep / 2 if t > 0 else 0.0, a - keep / 2 if a < total else total, "pause",
                               f"{a - t:.2f} sn duraklama"))
        t = max(t, b)
    return rem


def protect_words(removals: list[Removal], words: list[dict], removed_words: set[int]) -> list[Removal]:
    """Shrink pause/noise removals so they never clip a kept word."""
    kept = sorted((w["start"], w["end"]) for i, w in enumerate(words) if i not in removed_words)
    out = []
    for r in removals:
        pieces = [(r.start, r.end)]
        for ws, we in kept:
            nxt = []
            for a, b in pieces:
                if we <= a or ws >= b:
                    nxt.append((a, b))
                else:
                    if ws > a:
                        nxt.append((a, ws))
                    if we < b:
                        nxt.append((we, b))
            pieces = nxt
        for a, b in pieces:
            if b - a > 0.02:
                out.append(Removal(a, b, r.rule, r.reason, r.words))
    return out


def kept_segments(removals: list[Removal], total: float, min_len: float, words: list[dict]) -> list[tuple[float, float]]:
    cut = merge_regions([(r.start, r.end) for r in removals])
    segs = []
    t = 0.0
    for a, b in cut:
        if a > t:
            segs.append((t, a))
        t = max(t, b)
    if t < total:
        segs.append((t, total))
    out = []
    for a, b in segs:
        has_word = any(w["start"] < b and w["end"] > a for w in words)
        if b - a >= min_len or has_word:
            out.append((round(a, 4), round(b, 4)))
    return out


SENT_END = (".", "!", "?", "…")


def breathe_segments(segs: list[tuple[float, float]], words: list[dict], kept_words: list[dict], total: float,
                     tail_s: float = 0.18, tail_sentence_s: float = 0.32, head_s: float = 0.08, head_phrase_s: float = 0.12,
                     min_wordless_s: float = 0.6) -> list[tuple[float, float]]:
    """Give every kept piece room to breathe, like an editor trims by ear:
    - Whisper word ends are often early (the last syllable / vowel decay is after it), so a cut right at the word end
      clips the word ("hiccup"). Each piece keeps at least tail_s after its last word, tail_sentence_s after a
      sentence end, and head_s before its first word (head_phrase_s at the start of a phrase).
    - Never extends into the next/previous word (kept or removed), so fillers and retakes stay out.
    - Word-less crumbs shorter than min_wordless_s (noise flashes) are dropped."""
    if not segs:
        return segs
    starts = sorted(w["start"] for w in words)
    ends = sorted(w["end"] for w in words)
    import bisect
    out = []
    for a, b in segs:
        ws = [w for w in kept_words if a - 1e-3 <= (w["start"] + w["end"]) / 2 <= b + 1e-3]
        if not ws:
            if b - a >= min_wordless_s:
                out.append((a, b))
            continue
        last = max(ws, key=lambda w: w["end"])
        first = min(ws, key=lambda w: w["start"])
        final = str(last.get("word", "")).rstrip().endswith(SENT_END)
        j = bisect.bisect_right(starts, last["end"] - 1e-3)
        nxt = starts[j] if j < len(starts) else total
        new_b = max(b, min(last["end"] + (tail_sentence_s if final else tail_s), nxt - 0.04, total))
        i = bisect.bisect_left(ends, first["start"] + 1e-3) - 1
        prv = ends[i] if i >= 0 else 0.0
        head = head_phrase_s if (i < 0 or first["start"] - prv > 0.45) else head_s
        new_a = min(a, max(first["start"] - head, prv + 0.04, 0.0))
        out.append((new_a, new_b))
    out.sort()
    merged = []
    for a, b in out:
        if merged and a <= merged[-1][1] + 1e-3:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return [(round(a, 4), round(b, 4)) for a, b in merged]


# how sure a rule is, and whether it is applied without asking ("düşük" = only suggested)
CONFIDENCE = {"pause": "yüksek", "filler": "yüksek", "nonspeech": "orta", "long_wait": "orta", "repetition": "orta",
              "restart": "orta", "paraphrase": "düşük", "paraphrase_review": "düşük", "complete_kept": "düşük",
              "emphasis_kept": "düşük"}
WORD_RULES = {"filler", "repetition", "restart", "paraphrase", "paraphrase_review", "complete_kept", "emphasis_kept"}


def cut_key(rule: str, start: float) -> str:
    return f"{rule}@{start:.2f}"


def compute_cuts(words: list[dict], active: list[tuple[float, float]], total: float, cfg: dict,
                 protected: list[tuple[float, float]] | None = None,
                 paraphrase_sim: Callable[[str, str], float] | None = None,
                 extra: list[Removal] | None = None, overrides: dict | None = None) -> dict:
    fillers = set(cfg["fillers"])
    f_rem, f_idx = find_fillers(words, fillers)
    h_rem, h_idx = find_hallucinations(words, f_idx)
    f_rem, f_idx = f_rem + h_rem, f_idx | h_idx
    r_rem, r_idx = find_repetitions(words, cfg, f_idx, paraphrase_sim)
    removed = f_idx | r_idx
    n_rem = find_nonspeech(active, words, cfg["nonspeech_island_max_s"])
    w_rem = find_long_waits(words, removed, total, cfg["long_wait_min_s"], protected or [])
    p_rem = find_pauses(active, total, cfg["pause_cut_min_s"], cfg["pause_keep_s"])
    soft = protect_words(_outside(n_rem + p_rem + w_rem, protected or []), words, removed)
    all_rem = f_rem + r_rem + soft + list(extra or [])
    all_rem.sort(key=lambda r: r.start)
    notes = getattr(find_repetitions, "notes", [])
    ov = overrides or {}
    review, final = [], []
    for r, is_note in [(r, False) for r in all_rem] + [(n, True) for n in notes]:
        conf = CONFIDENCE.get(r.rule, "orta")
        key = cut_key(r.rule, r.start)
        default = (not is_note) and conf != "düşük"
        applied = {"keep": False, "cut": True}.get(ov.get(key), default)
        review.append(dict(r.to_dict(), key=key, confidence=conf, applied=applied, suggested=not default,
                           overridden=key in ov))
        if applied:
            final.append(r)
    final.sort(key=lambda r: r.start)
    word_cuts = [(r.start, r.end) for r in final if r.rule in WORD_RULES]
    removed = {i for i, w in enumerate(words)
               if any(a - 1e-3 <= (w["start"] + w["end"]) / 2 <= b + 1e-3 for a, b in word_cuts)}
    kept_words = [w for i, w in enumerate(words) if i not in removed]
    segs = kept_segments(final, total, cfg["min_kept_segment_s"], kept_words)
    if cfg.get("breathe", True):
        segs = breathe_segments(segs, words, kept_words, total, cfg.get("tail_s", 0.18), cfg.get("tail_sentence_s", 0.32),
                                cfg.get("head_s", 0.08), cfg.get("head_phrase_s", 0.12), cfg.get("min_wordless_s", 0.6))
    return {"removals": [r.to_dict() for r in final], "review": review, "kept": segs,
            "notes": [r.to_dict() for r in notes],
            "removed_word_idx": sorted(removed), "kept_duration": sum(b - a for a, b in segs), "source_duration": total}
