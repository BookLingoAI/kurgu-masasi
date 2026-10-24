"""A live board meeting: the same steps as meeting.hold_meeting, but spoken like a real conversation, and the user can
break in through the microphone.

Lines become events ({id, who, to, text, audio, phase}) that the room page (kurgu/web/kurul.html) plays in order.
Every spoken line is phrased by the agent itself from the last few lines of the conversation (names, reactions,
short spoken sentences); what it stands for - its finding, its vote - is fixed by code before it speaks, so the
wording can never change a decision. Voices are made in a separate thread while the next agent thinks; the room
only receives lines whose voice is ready, in order.
The language model runs one call at a time and pauses while a video job runs (both need the GPU).
"""
from __future__ import annotations

import json
import queue
import re
import threading
import time
from pathlib import Path

from . import agents as A
from . import meeting as MT
from .measure import AGENTS, measure
from .replay import speakable
from .voice import VOICES, say

NAMES = {VOICES[a]["isim"].lower().replace("İ", "i"): a for a in AGENTS}
FIELD_WORDS = {"ses": ("ses", "müzik", "efekt", "yüksek", "kısık", "lufs", "gürültü"),
               "renk": ("renk", "ışık", "karanlık", "koyu", "parlak", "pozlama", "netlik"),
               "grafik": ("yazı", "başlık", "metin", "font", "okun"),
               "animasyon": ("animasyon", "hareket", "donuk", "kıpırda", "canlı"),
               "gecis": ("kesim", "geçiş", "zıpla", "ritim", "zoom", "premiere")}
NO = re.compile(r"(istemiyorum|beğenmedim|olmasın|yapma|gerek yok|reddediyorum|katılmıyorum|hayır|vazgeç|kalsın)", re.I)
YES = re.compile(r"(onaylıyorum|katılıyorum|evet|güzel fikir|uygula|yapalım|harika|tamam yap)", re.I)
STYLE = ("KONUŞMA ÜSLUBU: Bir toplantı masasında yüksek sesle konuşuyorsun. Konuşma dili kullan, kısa cümleler kur, "
         "en fazla iki cümle ve 32 kelime. Gerekirse karşındakine adıyla hitap et; önceki konuşmacıya gerçekten cevap ver "
         "(katıl, karşı çık, ekle). 'Bence', 'açıkçası', 'bak', 'şöyle' gibi doğal ifadeler kullanabilirsin ama abartma. "
         "Sayıları yuvarla ve sözle söyle (2,47 değil 'iki buçuk'). Madde işareti, başlık, tırnak, kod, JSON YOK. "
         "Ölçülmüş olanı 'ölçtük', karede gördüğünü 'karede görüyorum', tahminini 'tahminim' diye ayır. "
         "Hiçbir şeyin uygulandığını söyleme. Cümleye 'Bak' diye başlama; masada az önce kullanılan girişleri tekrarlama. "
         "Güven yüzdesini yalnız düşükse (yüzde yetmişin altında) söyle. Zaman için 't=' yazma, 'on birinci saniyede' de. "
         "İnsan gibi konuş (Humanizer kuralları): 'X değil, Y' karşıtlığı, 'asıl mesele / işin özü' gibi derin görünen "
         "laflar, zorlama üçlü sıralama ve tek cümlelik dramatik kapanış kullanma. "
         "Sadece söyleyeceğin cümleleri yaz.")
TABLE = ", ".join(f"{VOICES[a]['isim']} ({A.ROLES[a]['ad']})" for a in AGENTS)


def _identity(agent: str) -> str:
    """Who you are at the table - the first meeting showed Zeynep addressing herself and Selin praising Selin."""
    return (f"Senin adın {VOICES[agent]['isim']}; sen {A.ROLES[agent]['ad']} uzmanısın. Masadakiler: {TABLE}. "
            f"Kendinden üçüncü kişi olarak söz etme, kendine adınla hitap etme.")


def addressee(text: str) -> str | None:
    low = text.lower().replace("İ", "i")
    for name, a in NAMES.items():
        if re.search(rf"\b{name}", low):
            return a
    for a, ws in FIELD_WORDS.items():
        if any(w in low for w in ws):
            return a
    return None


class LiveMeeting:
    def __init__(self, proj, llm, voice_dir: Path | None = None, warm: bool = False, busy=None):
        self.proj, self.llm, self.warm = proj, llm, warm
        self.busy = busy or (lambda: None)          # -> name of a running video job, or None
        self.poll_s = 5.0
        self.voice_dir = voice_dir or (proj.dir / "kurul" / "ses")
        self.events: list[dict] = []
        self.inbox: queue.Queue = queue.Queue()
        self.state, self.thinking, self.error = "hazır", None, None
        self.findings: list[dict] = []
        self.ruled: list[dict] = []
        self.caps: dict = {}
        self.profile: dict = {}
        self._lock = threading.Lock()
        self._tts: queue.Queue = queue.Queue()
        self._tts_thread = threading.Thread(target=self._voice_worker, name="kurul-ses", daemon=True)
        self._tts_thread.start()
        self.thread: threading.Thread | None = None
        self.mems = {a: A.load_memory(Path(proj.cfg.root), a) for a in AGENTS}
        self.rep: dict = {}

    # ---- events and voices ----------------------------------------------------------------------------------
    def emit(self, who: str, text: str, to: str | None = None, phase: str = "", **extra) -> dict:
        silent = extra.pop("silent", False)
        with self._lock:
            ev = dict({"id": len(self.events), "who": who, "to": to, "text": text, "audio": None, "phase": phase,
                       "t": time.time(), "hazir": silent or who not in VOICES or not text.strip()}, **extra)
            self.events.append(ev)
        if not ev["hazir"]:
            self._tts.put(ev)
        return ev

    def _voice_worker(self) -> None:
        while True:
            ev = self._tts.get()
            try:
                ev["audio"] = say(ev["who"], speakable(ev["text"]), self.voice_dir).name
            except Exception:  # noqa: BLE001 - shown without voice rather than not at all
                ev["audio"] = None
            ev["hazir"] = True

    def wait_voices(self, timeout: float = 120.0) -> None:
        end = time.time() + timeout
        while time.time() < end and any(not e["hazir"] for e in self.events):
            time.sleep(0.1)

    def since(self, after: int) -> dict:
        """Events after `after` - only up to the first one whose voice is not ready, so the room plays them in order."""
        out = []
        with self._lock:
            for e in self.events:
                if e["id"] <= after:
                    continue
                if not e["hazir"]:
                    break
                out.append({k: v for k, v in e.items() if k != "hazir"})
        return {"events": out, "durum": self.state, "dusunen": self.thinking, "hata": self.error}

    # ---- the model, one call at a time, never next to a video job ------------------------------------------------
    def _gpu_free(self) -> None:
        job = self.busy()
        if not job:
            return
        before = self.state
        self.state = "video işi bekleniyor"
        while self.busy():
            time.sleep(self.poll_s)
        self.state = before

    def _ask(self, agent: str, fn, *a, **kw):
        self._gpu_free()
        self.thinking = agent
        try:
            return fn(*a, **kw)
        finally:
            self.thinking = None

    def _recent(self, n: int = 5) -> str:
        with self._lock:
            lines = [e for e in self.events if e["text"] and e["text"] != "(çekimser)"][-n:]
        name = lambda w: VOICES[w]["isim"] if w in VOICES else ("Sen (yönetmen)" if w == "sen" else w)  # noqa: E731
        return "\n".join(f"{name(e['who'])}: {e['text']}" for e in lines) or "(henüz kimse konuşmadı)"

    def _line(self, agent: str, task: str, fallback: str, to: str | None = None) -> str:
        """The agent phrases what it has to say, having heard the last lines. Falls back to the plain sentence."""
        who_to = VOICES[to]["isim"] if to in VOICES else ("yönetmene (kullanıcıya)" if to == "sen" else "masaya")
        try:
            text = self._ask(agent, self.llm.chat, MT._system(agent) + " " + _identity(agent) + " " + STYLE,
                             f"Masada son konuşulanlar:\n{self._recent()}\n\nŞimdi sen konuşuyorsun, {who_to} hitap ederek.\n"
                             f"Söylemen gereken: {task}", max_tokens=110, temperature=0.75)
            text = _one_para(text)
        except Exception:  # noqa: BLE001
            text = ""
        return text if 3 <= len(text.split()) <= 60 else fallback

    # ---- what kind of video: the user's choice, or a guess the chair checks with the user when it is weak --------------
    def _settle_profile(self, wait: float = 45.0) -> dict:
        from . import profiles as PR
        prof = PR.resolve(self.proj)
        if not prof.get("sor"):
            return prof
        p = PR.PROFILES[prof["profil"]]
        self.emit(A.CHAIR, f"Bir şey sormam lazım: bu videoyu {p['ad'].lower()} videosu gibi görüyorum ama emin değilim. "
                  "Türünü sağ üstteki listeden seçer misin, ya da söyler misin? Biraz bekliyorum.", to="sen", phase="profil",
                  soru="profil")
        names = {PR.norm_word(v["ad"].split()[0]): k for k, v in PR.PROFILES.items()}
        end = time.time() + wait
        while time.time() < end:
            cur = PR.load(self.proj)
            if cur.get("kaynak") == "kullanici":
                return PR.resolve(self.proj)
            try:
                text = self.inbox.get(timeout=1.0)
            except queue.Empty:
                continue
            hit = next((k for w in text.split() for n, k in names.items() if PR.norm_word(w).startswith(n)), None)
            if hit:
                return PR.set_by_user(self.proj, hit)
            self._reply(text)
        self.emit(A.CHAIR, f"Cevap gelmedi; şimdilik {p['ad'].lower()} diye devam ediyorum, sonra değiştirebilirsin.",
                  to="sen", phase="profil")
        return prof

    # ---- the user breaks in ---------------------------------------------------------------------------------
    def user_says(self, text: str) -> None:
        text = text.strip()
        if text:
            self.emit("sen", text, to=addressee(text))
            self.inbox.put(text)

    def _answer_user(self, wait: float = 0.0) -> None:
        end = time.time() + wait
        while True:
            try:
                text = self.inbox.get(timeout=max(0.0, end - time.time())) if wait else self.inbox.get_nowait()
            except queue.Empty:
                return
            self._reply(text)
            end = time.time() + wait

    def _reply(self, text: str) -> None:
        a = addressee(text) or self._last_speaker() or A.CHAIR
        mem = self.mems[a]
        own = [f for f in (self.ruled or self.findings) if f["ajan"] == a]
        target = own[-1] if own else None
        # speech recognition can mishear: a spoken yes/no is only a CANDIDATE lesson until confirmed with a button
        if NO.search(text) or YES.search(text):
            A.add_lesson(mem, f"Kullanıcı {'karşı çıktı' if NO.search(text) else 'onayladı'}"
                              f"{' (' + target['baslik'] + ')' if target else ''}: \"{text[:160]}\"",
                         "kullanici_sesli", scope=f"proje:{self.proj.name}", confidence=0.5, status="aday")
            A.save_memory(Path(self.proj.cfg.root), a, mem)
        ctx = "\n".join(f"- {f['baslik']} ({f.get('karar', 'görüşülüyor')})" for f in (self.ruled or self.findings)) or "- (henüz öneri yok)"
        mine = "\n".join(f"- {f['baslik']}: kanıt {f['kanit']}" for f in own) or "- (benim önerim yok)"
        try:
            ans = self._ask(a, self.llm.chat, MT._system(a) + " " + _identity(a) + " " + STYLE + " Kullanıcı videonun sahibi ve yönetmendir.",
                            f"Masada son konuşulanlar:\n{self._recent()}\n\nKurulun önerileri:\n{ctx}\n\nSenin önerilerin:\n{mine}\n\n"
                            f"Yönetmen sana şunu söyledi: \"{text}\"\nOna doğrudan cevap ver. Karşı çıktıysa tartışma, notunu aldığını "
                            "ve kalıcı olması için sonda kararlar ekranından onaylaması gerektiğini söyle.", max_tokens=110, temperature=0.6)
            ans = _one_para(ans) or "Seni duydum, not aldım."
        except Exception as e:  # noqa: BLE001
            ans = f"Şu an cevap veremiyorum: {str(e)[:60]}"
        self.emit(a, ans, to="sen", phase="soru-cevap")

    def _last_speaker(self) -> str | None:
        with self._lock:
            return next((e["who"] for e in reversed(self.events) if e["who"] in AGENTS), None)

    # ---- the meeting -----------------------------------------------------------------------------------------
    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="kurul", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            self._meeting()
            self.state = "bitti"
            self.emit(A.CHAIR, "Toplantıyı kapatıyorum. Sormak istediğin bir şey varsa mikrofona konuş, on dakika daha buradayız. "
                      "Kararları sağdaki listeden tek tek onaylayabilir ya da reddedebilirsin.", to="sen", phase="soru-cevap")
            self._answer_user(wait=600.0)
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"[:300]
            self.state = "hata"
            self.emit(A.CHAIR, "Toplantı bir hata yüzünden durdu. Ayrıntı ekranın üstünde.", to="sen", phase="hata")
        finally:
            self.thinking = None
            if self.state != "hata":
                self.state = "kapandı"

    def _meeting(self) -> None:
        from .model import capabilities
        from .premiere_view import attach as attach_premiere
        proj = self.proj
        root = Path(proj.cfg.root)
        if self.warm:
            self.state = "model yükleniyor"
            self.emit(A.CHAIR, "Dil modeli yükleniyor, bu bir iki dakika sürebilir.", to="sen", phase="hazırlık")
            self._gpu_free()
            self.llm.start()
            self.llm.warm(900)
        self.caps = self._ask(A.CHAIR, capabilities, self.llm, root)
        self.state = "ölçülüyor"
        title = proj.settings.get("title") or proj.name
        self.emit(A.CHAIR, f"Hoş geldiniz arkadaşlar. Bugün {title} videosuna bakıyoruz. Ben Zeynep, toplantıyı ben yöneteceğim. "
                  "Önce herkes kendi alanında gördüğünü anlatsın, sonra tartışıp karar verelim.", to=None, phase="açılış")
        kdir = proj.dir / "kurul"
        rep_p = kdir / "olcum.json"
        video = (proj.load("kurgu.json") or {}).get("file")
        if not rep_p.exists() or (video and Path(video).exists() and Path(video).stat().st_mtime > rep_p.stat().st_mtime):
            self.rep = self._ask(A.CHAIR, measure, proj)
        else:
            self.rep = json.loads(rep_p.read_text(encoding="utf-8"))
        rep, settings = self.rep, proj.settings
        attach_premiere(proj, rep)                 # read-only Premiere timeline for Can, when the bridge answers
        self.profile = self._settle_profile()
        words = (proj.load("edit_plan.json") or {}).get("transcript_cut")
        if not self.caps.get("vision"):
            self.emit(A.CHAIR, "Bir not: modelimiz bugün kareleri göremiyor, o yüzden yalnızca ölçümlerle konuşacağız.",
                      to=None, phase="açılış")
        self._answer_user()
        # 1. views: findings are structured, then the agent tells the table in its own words
        self.state = "görüşler"
        for a in AGENTS:
            sheet = rep[a].get("kanit")
            imgs = [Path(sheet).read_bytes()] if self.caps.get("vision") and sheet and Path(sheet).exists() else []
            try:
                raw = self._ask(a, self.llm.see_json if imgs else _text_only(self.llm), MT._system(a),
                                MT._view_prompt(a, rep, self.mems[a], settings, vision=bool(imgs), profile=self.profile), imgs,
                                MT.VIEW_SCHEMA, max_tokens=1000)
                got = MT._findings(a, raw, rep, words, self.mems[a])
            except Exception:  # noqa: BLE001
                got = []
            self.findings += got
            valid = sorted((f for f in got if not f.get("gecersiz")), key=lambda f: f["oncelik"])
            phase = f"görüşler · {VOICES[a]['isim']}"
            if valid:
                f = valid[0]
                plain = f"{f['baslik']}. {f['kanit']} Önerim: {A.ACTIONS[f['eylem']][1]}."
                task = (f"Kendi alanındaki en önemli bulguyu masaya anlat: '{f['baslik']}'. Kanıtın: {f['kanit']}. "
                        f"Önerin: {A.ACTIONS[f['eylem']][1]}. Güvenin {f.get('guven', 0.7):.0%}." +
                        (f" Bir cümleyle ikinci konuyu da an: {valid[1]['baslik']}." if len(valid) > 1 else ""))
                self.emit(a, self._line(a, task, plain, to=None), to=None, phase=phase, finding=f["id"])
            elif got:
                self.emit(a, self._line(a, f"Bir önerin vardı ama kurallara uymadığı için sayılmadı ({got[0]['gecersiz']}). "
                                           "Bunu dürüstçe ve kısaca söyle.", "Bir önerim vardı ama kurallara uymadığı için sayılmadı.",
                                        to=A.CHAIR), to=A.CHAIR, phase=phase)
            else:
                self.emit(a, self._line(a, "Kendi alanında önemli bir sorun görmediğini, kısaca neden olduğunu söyle.",
                                        "Benim alanımda önemli bir sorun görmedim."), to=None, phase=phase)
            self._answer_user()
        valid = [f for f in self.findings if not f.get("gecersiz")]
        # 2. votes (structured, silent), then the debates aloud
        self.state = "oylama"
        votes: dict[str, dict] = {}
        for a in AGENTS:
            others = [f for f in valid if f["ajan"] != a]
            if not others:
                continue
            try:
                raw = self._ask(a, self.llm.chat_json, MT._system(a), MT._vote_prompt(a, rep, others), MT.VOTE_SCHEMA,
                                max_tokens=150 + 60 * len(others), temperature=0.2)
                votes[a] = {str(v.get("id")): v for v in raw.get("oylar") or [] if isinstance(v, dict)}
            except Exception:  # noqa: BLE001
                pass
        self.ruled = MT.rule(self.findings, votes)
        self._answer_user()
        self.state = "tartışma"
        debated = sorted((f for f in self.ruled if f["karar"] != "gecersiz"),
                         key=lambda f: (-(f["itiraz"] > 0), -f["itiraz"], -f["destek"]))[:3]
        for f in debated:
            name = VOICES[f["ajan"]]["isim"]
            phase = f"tartışma · {name}"
            self.emit(f["ajan"], self._line(f["ajan"], f"Masaya şu önerini tartışmaya aç ve fikirlerini sor: {f['baslik']}.",
                                            f"{f['baslik']} konusunda ne düşünüyorsunuz?"), phase=phase, clearVotes=True)
            objections = []
            order = sorted((v for v in f["oylar"] if v != f["ajan"]),
                           key=lambda v: {"itiraz": 0, "destek": 1}.get(f["oylar"][v]["oy"], 2))
            for voter in order:
                v = f["oylar"][voter]
                if v["oy"] == "cekimser" or not v["neden"]:
                    self.emit(voter, "(çekimser)", to=f["ajan"], phase=phase, votes={voter: "cekimser"}, pause=150, silent=True)
                    continue
                stance = "KARŞI ÇIKIYORSUN" if v["oy"] == "itiraz" else "DESTEKLİYORSUN"
                said = self._line(voter, f"{name}'in '{f['baslik']}' önerisini {stance}. Gerekçen: {v['neden']}. "
                                         "Bu tutumundan şaşma; önceki konuşmacıya da cevap verebilirsin.",
                                  f"{name}, {v['neden']}", to=f["ajan"])
                self.emit(voter, said, to=f["ajan"], phase=phase, votes={voter: v["oy"]})
                if v["oy"] == "itiraz":
                    objections.append((voter, v["neden"]))
                self._answer_user()
            if objections:     # the proposer answers, looking at the first one who objected
                first = objections[0][0]
                self.emit(f["ajan"], self._line(f["ajan"], "İtirazlara cevap ver: ya ikna olduğunu söyle ya da kanıtınla savun. "
                                                f"Kanıtın: {f['kanit']}. İtirazlar: " +
                                                "; ".join(f"{VOICES[o]['isim']}: {r}" for o, r in objections),
                                                "Kanıtım ölçümde, ama itirazları not aldım.", to=first), to=first, phase=phase)
            verdict = {"kabul": "kabul ediyoruz", "red": "bu haliyle geçmedi", "veto": "alan sahibinin itirazıyla duruyor",
                       "cakisma": "başka bir öneriyle çakıştığı için bekliyor", "gelistirici_isi": "geliştiriciye iletiyoruz"}[f["karar"]]
            rule_txt = (f"kabul için en az {MT.PASS_SUPPORT} destek ve en fazla bir itiraz gerekiyor; 'etkilemez' diyen "
                        "destekler çekimser sayılıyor")
            self.emit(A.CHAIR, self._line(A.CHAIR, f"Bu konuyu kapat. Oy sonucu: {f['destek']} destek, {f['itiraz']} itiraz; "
                                                   f"karar: öneri {verdict}. Kural: {rule_txt}. Kararı ve nedenini kısaca söyle.",
                                          f"{f['destek']} destek, {f['itiraz']} itiraz. Öneri {verdict}.", to="sen"),
                      to="sen", phase="karar")
            self._answer_user()
        # 3. ruling, memory, minutes
        self.state = "kararlar"
        passed = [f for f in self.ruled if f["karar"] in ("kabul", "gelistirici_isi")]
        bad = [f for f in self.ruled if f["karar"] == "gecersiz"]
        titles = "; ".join(f["baslik"] for f in passed[:5]) or "hiçbiri"
        self.emit(A.CHAIR, f"Özetliyorum: {len(passed)} öneri kabul edildi: {titles}. " +
                  (f"{len(bad)} öneriyi kurallara uymadığı için saymadım. " if bad else "") +
                  "Hiçbiri uygulanmadı, hepsi senin onayını bekliyor.", to="sen", phase="kararlar", clearVotes=True)
        for f in self.ruled:
            A.record(self.mems[f["ajan"]], f["eylem"] or "?", "oneri")
            if f["karar"] in ("kabul", "gelistirici_isi"):
                A.record(self.mems[f["ajan"]], f["eylem"], "kurul_kabul")
        for a in AGENTS:
            self.mems[a]["toplanti"] = self.mems[a].get("toplanti", 0) + 1
            A.save_memory(root, a, self.mems[a])
        # resource requests go to the one package manager (local library only; it never downloads)
        from .packages import request as resource_request
        requests = []
        for f in self.ruled:
            if f["eylem"] == "kaynak_iste" and f["karar"] in ("kabul", "gelistirici_isi"):
                req = dict(f["parametre"], ajan=f["ajan"], profil=(self.profile or {}).get("profil"))
                ans = resource_request(root, req)
                if ans["durum"] == "eksik" and not ans.get("lisans_bekleyen"):
                    from .fetch import satisfy          # approved sources only, verified before use
                    ans["indirme"] = satisfy(root, req)
                requests.append({"bulgu": f["id"], "istek": req, "cevap": ans})
        self.wait_voices()
        with self._lock:
            talk = [{k: e[k] for k in ("who", "to", "text", "phase")} for e in self.events if e["text"]]
        minutes = {"proje": proj.name, "tarih": time.strftime("%Y-%m-%d %H:%M"), "mod": "oner", "canli": True,
                   "model": getattr(self.llm, "key", None), "yetenek": self.caps, "bulgular": self.ruled, "konusmalar": talk,
                   "olcum": {a: rep[a]["metrics"] for a in AGENTS}, "eksik": rep.get("missing"), "turlar": {},
                   "premiere": rep.get("gecis", {}).get("premiere"), "profil": self.profile, "kaynak_istekleri": requests,
                   "sure_sn": round(time.time() - self.events[0]["t"], 1)}
        kdir.mkdir(exist_ok=True)
        (kdir / "tutanak.json").write_text(json.dumps(minutes, ensure_ascii=False, indent=1), encoding="utf-8")
        (kdir / "tutanak.md").write_text(MT.minutes_md(minutes), encoding="utf-8")
        if getattr(proj, "log", None):
            proj.log.write("kurul", "toplanti", f"Kurgu Kurulu: {len(passed)} kabul, {len(bad)} geçersiz, mod öner")


def _text_only(llm):
    """see_json's signature for a model that cannot see: the images are dropped, the prompt says so."""
    def call(system, user, imgs, schema, **kw):
        return llm.chat_json(system, user, schema, **kw)
    return call


def _one_para(text: str) -> str:
    text = re.sub(r"[*#`>\[\]{}]", "", text or "")
    text = re.sub(r"\bt\s*=\s*(\d+(?:[.,]\d+)?)(?:\s*(?:saniyede|saniye|sn)\b)?", lambda m: f"{m.group(1)}. saniyede", text)
    text = re.sub(r"\s+", " ", text.strip().strip('"').strip("“”"))
    return text[:320]
