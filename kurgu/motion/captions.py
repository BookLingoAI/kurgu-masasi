"""Word-by-word captions for vertical shorts as an ASS subtitle file (burned in by ffmpeg/libass)."""
from __future__ import annotations

from ..analysis.text import tr_lower


def tr_upper(s: str) -> str:
    return s.replace("i", "İ").replace("ı", "I").upper()


def _ts(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _clean(w: str) -> str:
    return w.strip().strip(",;:").replace("...", "").replace("…", "")


def group_words(words: list[dict], max_words: int = 3, max_chars: int = 18, max_gap: float = 0.35) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for w in words:
        if not _clean(w["word"]):
            continue
        g = groups[-1] if groups else None
        if (g and len(g) < max_words and sum(len(_clean(x["word"])) + 1 for x in g) + len(_clean(w["word"])) <= max_chars
                and w["start"] - g[-1]["end"] <= max_gap and not g[-1]["word"].rstrip().endswith((".", "?", "!"))):
            g.append(w)
        else:
            groups.append([w])
    return groups


def displayable(w: dict) -> bool:
    """Words worth showing: not transcription garbage ("suççççç…", ",7", invented "abone ol" over music)."""
    import re as _re
    t = _re.sub(r"[^\wçğıöşüÇĞİÖŞÜ]", "", w.get("word", ""))
    if not t or len(t) > 22 or _re.search(r"(.)\1{3,}", t.lower()):
        return False
    if t.isdigit() and len(t) <= 2 and not _re.search(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]", w.get("word", "")) and \
            _re.match(r"^\W", w.get("word", "")):
        return False         # ",7" pieces of a mis-heard number
    p = w.get("prob")
    return not (p is not None and p < 0.56 and t.lower() in {"abone", "ol", "olun", "altyazı", "mk", "izlediğiniz"})


def build_ass(words: list[dict], width: int, height: int, font: str = "Bahnschrift", size: int = 92,
              y_frac: float = 0.70, active_color: str = "&H0000E5FF", text_color: str = "&H00FFFFFF",
              animate: bool = False, max_words: int = 3, breaks: list[float] | None = None, style: str = "vurgu") -> str:
    """ASS with each group shown while spoken; the word being said is highlighted. style: "vurgu" (coloured word),
    "kutu" (the word on a coloured box: a thick outline in the accent colour, dark letters), "zipla" (coloured
    word that pops, group pops in)."""
    animate = animate or style == "zipla"
    try:   # dark letters on a light box, white letters on a dark one
        bb, gg, rr = (int(active_color[i:i + 2], 16) for i in (4, 6, 8))
        box_text = "&H00141414&" if 0.299 * rr + 0.587 * gg + 0.114 * bb > 147 else "&H00FFFFFF&"
    except ValueError:
        box_text = "&H00141414&"
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},{size},{text_color},{text_color},&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,7,3,5,60,60,0,1
Style: Tr,{font},{int(size * 0.78)},{text_color},{text_color},&H00000000,&H64000000,-1,-1,0,0,100,100,0,0,1,6,3,5,80,80,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    x = width // 2
    y = int(height * y_frac)
    lines = []
    english = [w for w in words if w.get("lang") == "en" and displayable(w)]
    groups = group_words([w for w in words if displayable(w) and w.get("lang") != "en"], max_words=max_words)
    for gi, g in enumerate(groups):
        g_end = groups[gi + 1][0]["start"] if gi + 1 < len(groups) else g[-1]["end"] + 0.3
        g_end = min(g_end, g[-1]["end"] + 0.6)
        for b in breaks or []:   # never run into a montage / scene that has no talk
            if g[0]["start"] < b < g_end:
                g_end = max(g[-1]["end"], b - 0.02)
        for k, w in enumerate(g):
            a = w["start"] if k else g[0]["start"]
            b = g[k + 1]["start"] if k + 1 < len(g) else g_end
            if b <= a:
                continue
            parts = []
            for j, x_w in enumerate(g):
                t = tr_upper(_clean(x_w["word"]))
                if j == k and style == "kutu":
                    box = max(6, size // 7)
                    parts.append("{\\c" + box_text + "\\3c" + active_color + f"\\bord{box}\\shad0" + "}" + t +
                                 "{\\c" + text_color + "\\3c&H00000000&\\bord7\\shad3}")
                elif j == k:
                    pop = "\\fscx112\\fscy112" if animate else ""
                    parts.append("{\\c" + active_color + pop + "}" + t + "{\\c" + text_color + "\\fscx100\\fscy100}")
                else:
                    parts.append(t)
            # the group pops in when it appears (first word), then stays steady
            intro = "\\fscx70\\fscy70\\t(0,110,\\fscx100\\fscy100)" if animate and k == 0 else ""
            lines.append(f"Dialogue: 0,{_ts(a)},{_ts(b)},Cap,,0,0,0,,{{\\pos({x},{y}){intro}}}" + " ".join(parts))
    # English (the speaker's wife, the waiters): the Turkish translation as one italic line for the whole sentence
    runs: list[list[dict]] = []
    for w in english:
        if runs and w["start"] - runs[-1][-1]["end"] < 0.8 and w.get("tr_line") == runs[-1][-1].get("tr_line"):
            runs[-1].append(w)
        else:
            runs.append([w])
    for r in runs:
        a, b = r[0]["start"], max(r[-1]["end"] + 0.5, r[0]["start"] + 1.6)
        for br in breaks or []:
            if a < br < b:
                b = max(r[-1]["end"], br - 0.02)
        text = r[0].get("tr_line") or " ".join(_clean(x["word"]) for x in r)
        text = text.replace("{", "(").replace("}", ")")
        lines.append(f"Dialogue: 0,{_ts(a)},{_ts(b)},Tr,,0,0,0,,{{\\pos({x},{y})\\c{active_color}}}» {{\\c{text_color}}}{text}")
    return header + "\n".join(lines) + "\n"


__all__ = ["build_ass", "group_words", "tr_upper", "tr_lower"]
