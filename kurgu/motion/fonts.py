"""Fonts for on-screen titles.

Professional free fonts (Google Fonts, SIL Open Font License) are used when the font package is installed
(<root>/assets/fonts, job "install_assets"); otherwise the best Windows font for the role. A font is only used if it
has every Turkish letter (ğ ş ı İ ö ü ç), so no title ever shows empty boxes.

Roles: "heavy" - big condensed display (prices, punch words), "bold" - headline text, "label" - small caps labels.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

FONT_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"
TURKISH = "ĞğŞşİıÖöÜüÇç"

# (file in FONT_DIR, variable-font weight or None) in order of preference
PRO = {
    "heavy": [("Anton-Regular.ttf", None), ("BebasNeue-Regular.ttf", None), ("Oswald[wght].ttf", 700)],
    "bold": [("Montserrat[wght].ttf", 800), ("Poppins-ExtraBold.ttf", None), ("Poppins-Bold.ttf", None)],
    "label": [("Oswald[wght].ttf", 600), ("Montserrat[wght].ttf", 700), ("Poppins-Bold.ttf", None)],
    # editorial look (kicker in spaced mono, a huge grotesk word, an italic serif accent line)
    "display": [("Inter[opsz,wght].ttf", (32, 820)), ("Montserrat[wght].ttf", 800)],
    "italic": [("InstrumentSerif-Italic.ttf", None)],
    "mono": [("IBMPlexMono-Medium.ttf", None)],
    "text": [("Inter[opsz,wght].ttf", (24, 640)), ("Montserrat[wght].ttf", 650)],
}
# files the font package downloads (google/fonts repository, OFL)
PACKAGE = {
    "Anton-Regular.ttf": "ofl/anton/Anton-Regular.ttf",
    "BebasNeue-Regular.ttf": "ofl/bebasneue/BebasNeue-Regular.ttf",
    "Montserrat[wght].ttf": "ofl/montserrat/Montserrat%5Bwght%5D.ttf",
    "Poppins-ExtraBold.ttf": "ofl/poppins/Poppins-ExtraBold.ttf",
    "Oswald[wght].ttf": "ofl/oswald/Oswald%5Bwght%5D.ttf",
    "Inter[opsz,wght].ttf": "ofl/inter/Inter%5Bopsz,wght%5D.ttf",
    "InstrumentSerif-Italic.ttf": "ofl/instrumentserif/InstrumentSerif-Italic.ttf",
    "IBMPlexMono-Medium.ttf": "ofl/ibmplexmono/IBMPlexMono-Medium.ttf",
}
LICENSES = {"anton": "ofl/anton/OFL.txt", "bebasneue": "ofl/bebasneue/OFL.txt", "montserrat": "ofl/montserrat/OFL.txt",
            "poppins": "ofl/poppins/OFL.txt", "oswald": "ofl/oswald/OFL.txt", "inter": "ofl/inter/OFL.txt",
            "instrumentserif": "ofl/instrumentserif/OFL.txt", "ibmplexmono": "ofl/ibmplexmono/OFL.txt"}


def _system(role: str) -> list[tuple[Path, str | None]]:
    if os.name == "nt":
        fd = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        return {"heavy": [(fd / "impact.ttf", None), (fd / "bahnschrift.ttf", "Bold Condensed")],
                "bold": [(fd / "bahnschrift.ttf", "Bold"), (fd / "seguibl.ttf", None), (fd / "arialbd.ttf", None)],
                "label": [(fd / "bahnschrift.ttf", "SemiBold Condensed"), (fd / "segoeuib.ttf", None)],
                "display": [(fd / "seguibl.ttf", None), (fd / "arialbd.ttf", None)],
                "italic": [(fd / "georgiai.ttf", None), (fd / "timesi.ttf", None)],
                "mono": [(fd / "consola.ttf", None), (fd / "cour.ttf", None)],
                "text": [(fd / "segoeuib.ttf", None), (fd / "arialbd.ttf", None)]}[role]
    return [(Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"), None)]


def has_turkish_bytes(data: bytes) -> bool:
    """Font file content has every Turkish letter (checked in memory: no file handle stays open - on Windows an
    open handle made renaming the downloaded font fail)."""
    try:
        from io import BytesIO

        from fontTools.ttLib import TTFont
        f = TTFont(BytesIO(data))
        try:
            cmap = f.getBestCmap() or {}
        finally:
            f.close()
        return all(ord(ch) in cmap for ch in TURKISH)
    except Exception:  # noqa: BLE001 - cannot check: trust the font
        return True


@lru_cache(maxsize=64)
def has_turkish(path: str) -> bool:
    try:
        return has_turkish_bytes(Path(path).read_bytes())
    except OSError:
        return False


def installed() -> list[str]:
    return sorted(p.name for p in FONT_DIR.glob("*.ttf")) if FONT_DIR.exists() else []


@lru_cache(maxsize=128)
def font(role: str, size: int) -> ImageFont.FreeTypeFont:
    for name, wght in PRO.get(role, []):
        p = FONT_DIR / name
        if p.exists() and has_turkish(str(p)):
            f = ImageFont.truetype(str(p), size)
            if wght:
                try:
                    f.set_variation_by_axes(list(wght) if isinstance(wght, tuple) else [wght])
                except Exception:  # noqa: BLE001
                    pass
            return f
    for p, var in _system(role):
        if p.exists():
            f = ImageFont.truetype(str(p), size)
            if var:
                try:
                    names = [n.decode() if isinstance(n, bytes) else n for n in f.get_variation_names()]
                    pick = next((n for n in names if n.replace(" ", "").lower() == var.replace(" ", "").lower()), None)
                    if pick:
                        f.set_variation_by_name(pick)
                except Exception:  # noqa: BLE001
                    pass
            return f
    return ImageFont.load_default()
