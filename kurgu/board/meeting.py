"""A board meeting on a finished edit (mode "oner": nothing is changed, the board only proposes).

  1. views  - each agent, one after another on the local model, looks at its own measurements and evidence sheet
              and writes up to 4 findings, each with evidence and one action from the catalogue (agents.ACTIONS)
  2. votes  - each agent reads the others' findings and supports, objects or abstains, with a one-line reason
  3. ruling - plain code (the referee): a finding passes with >= 3 of 6 in support, at most one objection and no
              objection from the owner of that kind of action; clashing findings keep the better supported one
  4. minutes - kurul/tutanak.json + kurul/tutanak.md; every agent's memory counts what it proposed and what passed

Agents run one at a time (one model on one GPU); a silent or broken answer costs only that agent's turn.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from . import agents as A
from .measure import AGENTS, measure

VIEW_SCHEMA = {"type": "object", "properties": {"bulgular": {"type": "array", "items": {"type": "object", "properties": {
    "baslik": {"type": "string"}, "olcum": {"type": "string"}, "gozlem": {"type": "string"}, "yorum": {"type": "string"},
    "kanit": {"type": "string"}, "eylem": {"type": "string"}, "parametre": {"type": "object"},
    "oncelik": {"type": "integer"}, "guven": {"type": "number"}}}}}, "required": ["bulgular"]}
MIN_CONFIDENCE = 0.5      # below this the agent abstains instead of proposing
# what this user has ruled out for good (CLAUDE.md) - a finding that asks for it never reaches the vote
AGAINST_USER = re.compile(r"(emoji|nefes tan|rötuş|rotus|s-eğri|s eğri|sert kontrast|kontrastı artır|abone ses|beğeni ses|"
                          r"like ses|yorum ses|rnnoise)", re.I)
VOTE_SCHEMA = {"type": "object", "properties": {"oylar": {"type": "array", "items": {"type": "object", "properties": {
    "id": {"type": "string"}, "oy": {"type": "string"}, "neden": {"type": "string"}}}}}, "required": ["oylar"]}
PASS_SUPPORT = 3          # the user set it (22.10.2026): 4 let nothing pass once "does no harm" stopped counting


def _noop(f, m):
    pass


def _sec(v) -> float | None:
    """12.5 / "12,5" / "1:04.8" -> seconds."""
    if v is None or isinstance(v, (int, float)):
        return v
    s = str(v).strip().replace(",", ".")
    m = re.fullmatch(r"(\d+):(\d+(?:\.\d+)?)", s)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    try:
        return float(s)
    except ValueError:
        return None


def _system(agent: str) -> str:
    r = A.ROLES[agent]
    return (f"Sen 'Kurgu Kurulu'nun {r['ad']} uzmanısın: {r['uzmanlik']} Türkçe YouTube videolarını (kişisel vlog, "
            "masa başı sohbet) daha çok izlenir hale getirmek için çalışırsın. Yalnızca sana verilen ölçümlere ve karelere "
            "dayanırsın; ölçülmemiş ya da görülmemiş şey uydurmazsın. Kısa, net ve somut yazarsın. " + A.COMMON_RULES +
            (" " + r["ek"] if r.get("ek") else ""))


def _view_prompt(agent: str, rep: dict, mem: dict, settings: dict, vision: bool = True, profile: dict | None = None) -> str:
    from .bilgi import checklist
    from .profiles import prompt_block
    sec = rep[agent]
    prof = (prompt_block(profile, agent) + "\n\n") if profile and profile.get("profil") else ""
    own = {k: v for k, v in A.ACTIONS.items() if v[0] in (agent, "*")}
    acts = "\n".join(f"- {k}: {v[1]}; parametre {json.dumps(v[2], ensure_ascii=False)}" for k, v in own.items())
    flags = "\n".join(f"- {f['zaman']} ({f['t']} sn) [{f['tur']}] {f['aciklama']}" for f in sec["flags"][:8]) or "- (işaretli an yok)"
    lessons = A.lessons_text(mem) or "- (henüz yok)"
    pack = sec.get("paket")
    pack = f"EK KANIT PAKETİN:\n{json.dumps(pack, ensure_ascii=False)[:1500]}\n\n" if pack else ""
    track = A.track_record(mem) or "(henüz yok)"
    o = settings.get("options") or {}
    missing = ", ".join(rep.get("missing") or []) or "yok"
    return (f"VİDEO: {rep['duration_s'] / 60:.1f} dk, kurgu türü '{o.get('tur', '?')}', yoğunluk '{o.get('intensity', '?')}', "
            f"altyazı {'açık' if o.get('captions') else 'kapalı'}. Ölçülemeyenler: {missing}.\n\n"
            f"SENİN ÖLÇÜMLERİN:\n{json.dumps(sec['metrics'], ensure_ascii=False)[:1800]}\n\n"
            f"İŞARETLİ ANLAR:\n{flags}\n\n{pack}"
            + ("GÖRSEL: kanıt karesi (her karenin üstünde saati ve ne olduğu yazıyor).\n\n" if vision else
               "GÖRSEL YOK: bugün kare göremiyorsun; görsel gözlem yazma, yalnız ölçümlere dayan.\n\n") +
            f"GENEL UZMAN KONTROL LİSTEN (kaynaklı; 'yaygın uygulama' kesin kural değildir):\n{checklist(agent)}\n\n"
            f"{prof}"
            f"DERSLERİN (bunlara aykırı öneri yapma):\n{lessons}\n\nGEÇMİŞİN: {track}\n\n"
            f"KULLANABİLECEĞİN EYLEMLER:\n{acts}\n\n"
            "GÖREV: Bu videoda kendi alanındaki EN ÖNEMLİ en fazla 4 sorunu yaz (en etkilisi önce). Her biri için: "
            "'baslik' (kısa), 'olcum' (hangi ölçüm hangi değeri verdi - yalnız yukarıdaki sayılardan), 'gozlem' (karede "
            "ne gördün; görmediysen boş), 'yorum' (bunun izleyiciye etkisi - tahminse 'tahmin' de), 'kanit' (saat ve "
            "kaynak, tek satır), 'eylem' (yukarıdaki listeden biri), 'parametre' (zaman 't' saniye cinsinden sayı), "
            "'oncelik' (1 = en önemli, 3 = az), 'guven' (0-1; kanıtın ne kadar güçlü). Kanıtın zayıfsa öneri yapma, "
            "çekimser kal. Önerin kontrol listene ve derslerine uyuyor mu, başka bir uzmanın alanına zarar veriyor mu: "
            "yazmadan önce kendin denetle. "
            "Ölçüm zaten hedefteyse ya da sorun yoksa öneri yapma; boş liste yazmak doğru bir cevaptır. Ekrana yazılacak "
            "metin konuşmacının o anda söylediği kelimelerden AYNEN olmalı (örnek ya da tarif yazma). Sayılar makul olsun: "
            "pozlama ±1,5 EV, efekt ±12 dB, yazı süresi 2-8 sn, kesim en çok 30 sn. Sadece JSON döndür.")


def _vote_prompt(agent: str, rep: dict, others: list[dict]) -> str:
    lines = []
    for f in others:
        lines.append(f"[{f['id']}] ({A.ROLES[f['ajan']]['ad']}) {f['baslik']} — eylem: {f['eylem']} "
                     f"{json.dumps(f['parametre'], ensure_ascii=False)} — kanıt: {f['kanit']}")
    sec = rep[agent]
    return (f"Kurulun diğer üyelerinin önerileri:\n" + "\n".join(lines) + "\n\n"
            f"Senin alanındaki ölçümler (bağlam için): {json.dumps(sec['metrics'], ensure_ascii=False)[:700]}\n\n"
            "Her öneri için oy ver: 'destek' (videoyu YouTube izleyicisi için iyileştirir ve senin alanına zarar vermez), "
            "'itiraz' (yanlış, gereksiz ya da senin alanına zarar verir) ya da 'cekimser' (emin değilsin / alanın değil). "
            "'Benim alanımı etkilemez' bir destek nedeni DEĞİLDİR: o durumda 'cekimser' de. "
            "Her oy için en fazla 15 kelimelik 'neden' yaz. Sadece JSON döndür: {\"oylar\": [{\"id\", \"oy\", \"neden\"}]}.")


# sane ranges: the first meeting asked for -9 EV (a black picture)
BOUNDS = {"pozlama": ("ev", -1.5, 1.5), "efekt_seviyesi": ("db", -12.0, 12.0), "yazi_suresi": ("sure_sn", 2.0, 8.0),
          "ses_seviyesi": ("hedef_lufs", -16.0, -12.0), "kesim_ayari": ("sn", 0.2, 30.0)}
PLACEHOLDER = re.compile(r"\b(ör|örn|örneğin|vb|xxx)\b\.?|\.\.\.|…|<[^>]+>|\(.*?\)", re.I)
NEUTRAL = re.compile(r"(etkilemez|etki(si)? yok|zarar vermez|değiştirmez|bozulmaz|ilgisi yok|ilgili değil|alan(ım)?(ın)? dışı|"
                     r"doğrudan (etki|ilişki|bağlant)|uzmanlığı(m)? dışı)", re.I)


def _check(f: dict, rep: dict | None, words: list[dict] | None) -> str | None:
    """Why a finding cannot stand (None = it can): placeholders, unsaid text, absurd numbers, no evidence."""
    act, par = f["eylem"], f["parametre"]
    for v in par.values():
        if isinstance(v, str) and PLACEHOLDER.search(v):
            return f"yer tutucu metin: '{v[:40]}'"
    if act in BOUNDS:
        key, lo, hi = BOUNDS[act]
        v = par.get(key)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return f"'{key}' sayı değil"
        if not lo <= v <= hi:
            return f"ölçüsüz değer: {key}={v:g} (izin verilen {lo:g}..{hi:g})"
    t = par.get("t")
    dur = (rep or {}).get("duration_s")
    if t is not None and dur and not 0 <= t <= dur:
        return f"zaman videonun dışında ({t:g} sn)"
    if act in ("yazi_ekle", "bolum_basligi") and words is not None:
        text = str(par.get("metin") or par.get("baslik") or "")
        if t is None or not text:
            return "zaman ya da metin yok"
        from ..editing.graphics import grounded
        near = {"words": [w for w in words if abs(w["t_start"] - t) <= 10]}
        if not near["words"] or not grounded(text, near):
            return f"'{text[:40]}' o anda söylenmemiş"
    if act == "ses_seviyesi" and rep:
        I = rep["ses"]["metrics"].get("I_LUFS")
        if I is not None and abs(I - float(par["hedef_lufs"])) < 1.0:
            return f"ses zaten hedefte ({I} LUFS)"
    if act == "kesim_ayari" and rep and t is not None:
        reach = 30.0 if par.get("islem") == "kisalt" else 6.0
        marks = [g for g in rep["hikaye"]["flags"] if g["tur"] in ("bosluk", "durgun", "tempo") and abs(g["t"] - t) <= reach]
        if not marks:
            return "kanıt yok: o anda ölçülmüş boşluk, durgunluk ya da tempo düşüşü yok"
    return None


def _confidence(v) -> float:
    try:
        x = float(str(v).replace("%", "").replace(",", "."))
    except (TypeError, ValueError):
        return 0.6           # not given: neither strong nor too weak
    return x / 100 if x > 1 else x


def _findings(agent: str, raw: dict, rep: dict | None = None, words: list[dict] | None = None,
              mem: dict | None = None) -> list[dict]:
    out = []
    for k, b in enumerate((raw.get("bulgular") or [])[:4]):
        if not isinstance(b, dict):
            continue
        act = str(b.get("eylem") or "").strip()
        par = b.get("parametre") if isinstance(b.get("parametre"), dict) else {}
        if "t" in par:
            par["t"] = _sec(par["t"])
        why = A.validate(act, par)
        owner = A.ACTIONS.get(act, ("?",))[0]
        if why is None and owner not in (agent, "*"):
            why = f"'{act}' eylemi {owner} ajanının alanı"
        f = {"id": f"{agent[:3].upper()}{k + 1}", "ajan": agent, "baslik": str(b.get("baslik") or "")[:120],
             "kanit": str(b.get("kanit") or "")[:200], "eylem": act, "parametre": par,
             "oncelik": int(b.get("oncelik") or 2) if str(b.get("oncelik") or "2").isdigit() else 2, "gecersiz": why,
             "olcum": str(b.get("olcum") or "")[:200], "gozlem": str(b.get("gozlem") or "")[:200],
             "yorum": str(b.get("yorum") or "")[:200], "guven": round(_confidence(b.get("guven")), 2)}
        if why is None:
            f["gecersiz"] = _check(f, rep, words)
        # the agent's own second look, done by code: weak evidence = abstain; never against the user's standing rules
        if f["gecersiz"] is None and f["guven"] < MIN_CONFIDENCE:
            f["gecersiz"] = f"düşük güven ({f['guven']:.0%}): çekimser kalındı"
        blob = " ".join([f["baslik"], f["kanit"], f["yorum"], json.dumps(par, ensure_ascii=False)])
        if f["gecersiz"] is None and act not in ("yazi_kaldir",) and AGAINST_USER.search(blob):   # removing it is fine
            f["gecersiz"] = f"kullanıcının kalıcı tercihine aykırı ('{AGAINST_USER.search(blob).group(0)}')"
        out.append(f)
    return out


def rule(findings: list[dict], votes: dict[str, dict[str, dict]]) -> list[dict]:
    """The referee. votes[voter][finding id] = {"oy", "neden"}. Returns the findings with tally and verdict."""
    out = []
    for f in findings:
        if f.get("gecersiz"):
            out.append(dict(f, karar="gecersiz", destek=0, itiraz=0, cekimser=0, oylar={}))
            continue
        tally = {f["ajan"]: {"oy": "destek", "neden": "öneren"}}
        for voter in AGENTS:
            if voter != f["ajan"]:
                v = votes.get(voter, {}).get(f["id"]) or {"oy": "cekimser", "neden": "oy yok"}
                oy = str(v.get("oy", "")).lower().replace("ç", "c")
                why = str(v.get("neden") or "")[:140]
                oy = oy if oy in ("destek", "itiraz", "cekimser") else "cekimser"
                if oy == "destek" and NEUTRAL.search(why):   # "it does not touch my field" is not a reason to back it
                    oy = "cekimser"
                tally[voter] = {"oy": oy, "neden": why}
        sup = sum(v["oy"] == "destek" for v in tally.values())
        obj = sum(v["oy"] == "itiraz" for v in tally.values())
        owner = A.ACTIONS[f["eylem"]][0]
        veto = owner in AGENTS and owner != f["ajan"] and tally.get(owner, {}).get("oy") == "itiraz"
        if veto:
            verdict = "veto"
        elif sup >= PASS_SUPPORT and obj <= 1:
            verdict = "gelistirici_isi" if f["eylem"] == "gelistirici" else "kabul"
        else:
            verdict = "red"
        out.append(dict(f, karar=verdict, destek=sup, itiraz=obj, cekimser=len(tally) - sup - obj, oylar=tally))
    # clashing proposals (same action at the same moment, different settings): the better supported one stays
    passed = sorted((f for f in out if f["karar"] == "kabul"), key=lambda f: (-f["destek"], f["itiraz"], f["oncelik"]))
    seen: list[dict] = []
    for f in passed:
        t = f["parametre"].get("t")
        clash = next((g for g in seen if g["eylem"] == f["eylem"] and g["parametre"] != f["parametre"] and
                      (t is None or g["parametre"].get("t") is None or abs(g["parametre"]["t"] - t) < 1.0)), None)
        if clash:
            f["karar"] = "cakisma"
            f["not"] = f"{clash['id']} daha çok destek aldı"
        else:
            seen.append(f)
    return out


def hold_meeting(proj, llm, progress=_noop, remeasure: bool = False) -> dict:
    root = Path(proj.cfg.root)
    kdir = proj.dir / "kurul"
    rep_p = kdir / "olcum.json"
    video = (proj.load("kurgu.json") or {}).get("file")
    if remeasure or not rep_p.exists() or (video and Path(video).exists() and Path(video).stat().st_mtime > rep_p.stat().st_mtime):
        rep = measure(proj, lambda f, m: progress(0.15 * f, m))
    else:
        rep = json.loads(rep_p.read_text(encoding="utf-8"))
    settings = proj.settings
    words = (proj.load("edit_plan.json") or {}).get("transcript_cut")
    mems = {a: A.load_memory(root, a) for a in AGENTS}
    t_start = time.time()
    findings, turns = [], {}
    for k, a in enumerate(AGENTS):
        progress(0.15 + 0.45 * k / len(AGENTS), f"{A.ROLES[a]['ad']} videoyu inceliyor")
        t0 = time.time()
        sheet = rep[a].get("kanit")
        imgs = [Path(sheet).read_bytes()] if sheet and Path(sheet).exists() else []
        try:
            raw = llm.see_json(_system(a), _view_prompt(a, rep, mems[a], settings), imgs, VIEW_SCHEMA, max_tokens=900)
            got = _findings(a, raw, rep, words)
            turns[a] = {"sn": round(time.time() - t0, 1), "durum": "tamam", "bulgu": len(got)}
        except Exception as e:  # noqa: BLE001 - this agent sits the round out
            got = []
            turns[a] = {"sn": round(time.time() - t0, 1), "durum": f"cevap yok: {str(e)[:120]}", "bulgu": 0}
        findings += got
    valid = [f for f in findings if not f.get("gecersiz")]
    votes: dict[str, dict[str, dict]] = {}
    for k, a in enumerate(AGENTS):
        progress(0.6 + 0.35 * k / len(AGENTS), f"{A.ROLES[a]['ad']} önerileri oyluyor")
        others = [f for f in valid if f["ajan"] != a]
        if not others:
            continue
        t0 = time.time()
        try:
            raw = llm.chat_json(_system(a), _vote_prompt(a, rep, others), VOTE_SCHEMA, max_tokens=150 + 60 * len(others),
                                temperature=0.2)
            votes[a] = {str(v.get("id")): v for v in raw.get("oylar") or [] if isinstance(v, dict)}
            turns[a]["oy_sn"] = round(time.time() - t0, 1)
        except Exception as e:  # noqa: BLE001
            turns[a]["oy_sn"] = round(time.time() - t0, 1)
            turns[a]["oy_durum"] = f"oy yok: {str(e)[:120]}"
    ruled = rule(findings, votes)
    for f in ruled:
        A.record(mems[f["ajan"]], f["eylem"] or "?", "oneri")
        if f["karar"] in ("kabul", "gelistirici_isi"):
            A.record(mems[f["ajan"]], f["eylem"], "kurul_kabul")
    for a in AGENTS:
        mems[a]["toplanti"] = mems[a].get("toplanti", 0) + 1
        A.save_memory(root, a, mems[a])
    minutes = {"proje": proj.name, "tarih": time.strftime("%Y-%m-%d %H:%M"), "mod": "oner", "sure_sn": round(time.time() - t_start, 1),
               "model": getattr(llm, "key", None), "turlar": turns, "bulgular": ruled,
               "olcum": {a: rep[a]["metrics"] for a in AGENTS}, "eksik": rep.get("missing")}
    kdir.mkdir(exist_ok=True)
    (kdir / "tutanak.json").write_text(json.dumps(minutes, ensure_ascii=False, indent=1), encoding="utf-8")
    (kdir / "tutanak.md").write_text(minutes_md(minutes), encoding="utf-8")
    progress(1.0, "Kurul toplantısı bitti")
    return minutes


VERDICT = {"kabul": "✅ KABUL", "gelistirici_isi": "🛠️ GELİŞTİRİCİ İŞİ", "red": "❌ red", "veto": "⛔ veto",
           "cakisma": "↔️ çakışma", "gecersiz": "⚠️ geçersiz"}
VOTE_MARK = {"destek": "👍", "itiraz": "👎", "cekimser": "·"}


def minutes_md(m: dict) -> str:
    lines = [f"# Kurgu Kurulu tutanağı — {m['proje']}", "",
             f"{m['tarih']} · mod: **sadece öner** (hiçbir şey uygulanmadı) · {m['sure_sn'] / 60:.1f} dk · model {m['model']}", ""]
    passed = [f for f in m["bulgular"] if f["karar"] in ("kabul", "gelistirici_isi")]
    lines += ["## Kurulun kararları", ""]
    if passed:
        for f in sorted(passed, key=lambda f: (f["oncelik"], -f["destek"])):
            lines.append(f"- {VERDICT[f['karar']]} **{f['baslik']}** ({A.ROLES[f['ajan']]['ad']}, {f['destek']}/6 destek) — "
                         f"`{f['eylem']}` {json.dumps(f['parametre'], ensure_ascii=False)}  \n  Kanıt: {f['kanit']}")
    else:
        lines.append("- Hiçbir öneri yeterli destek almadı.")
    lines += ["", "## Ajanların görüşleri ve oylar", ""]
    for a in AGENTS:
        t = m["turlar"].get(a, {})
        lines.append(f"### {A.ROLES[a]['ad']} — {t.get('durum', '?')} ({t.get('sn', 0)} sn görüş, {t.get('oy_sn', '-')} sn oy)")
        own = [f for f in m["bulgular"] if f["ajan"] == a]
        if not own:
            lines.append("- (öneri yok)")
        for f in own:
            lines.append(f"- **[{f['id']}] {f['baslik']}** → {VERDICT[f['karar']]}" +
                         (f" ({f['destek']} destek / {f['itiraz']} itiraz)" if f["karar"] != "gecersiz" else f": {f['gecersiz']}"))
            lines.append(f"  - Kanıt: {f['kanit']}")
            lines.append(f"  - Eylem: `{f['eylem']}` {json.dumps(f['parametre'], ensure_ascii=False)}")
            for voter, v in (f.get("oylar") or {}).items():
                if voter != f["ajan"]:
                    lines.append(f"  - {VOTE_MARK.get(v['oy'], '·')} {A.ROLES[voter]['ad']}: {v['neden']}")
            if f.get("not"):
                lines.append(f"  - Not: {f['not']}")
        lines.append("")
    return "\n".join(lines)
