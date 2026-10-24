"""Chunked transcription (section 5.2).

The audio is split at energy pauses into chunks (<= chunk_max_s) and each chunk is transcribed on its own, so
repeated takes are not skipped and word times do not drift. Short uncertain non-speech sounds are not counted as
words; Whisper's stock hallucinations are removed; names are corrected with dictionary.json.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .audio import active_regions, energy_profile
from .text import correct_words, is_hallucination, norm_word


def plan_chunks(speech: list[tuple[float, float]], max_len: float, min_pause: float, pad: float, total: float,
                min_chunk: float = 1.0) -> list[tuple[float, float]]:
    """One chunk per spoken phrase: split at every pause >= min_pause.

    Very short phrases (< min_chunk) are attached to a close neighbour for context; overly long continuous speech
    is hard-split at max_len."""
    chunks: list[list[float]] = []
    for a, b in speech:
        if chunks and a - chunks[-1][1] < min_pause:
            chunks[-1][1] = b
        else:
            chunks.append([a, b])
    merged: list[list[float]] = []
    for c in chunks:
        if merged and (c[1] - c[0] < min_chunk or merged[-1][1] - merged[-1][0] < min_chunk) \
                and c[0] - merged[-1][1] < 0.8 and c[1] - merged[-1][0] <= max_len:
            merged[-1][1] = c[1]
        else:
            merged.append(list(c))
    out = []
    for a, b in merged:
        while b - a > max_len:
            out.append((a, a + max_len))
            a += max_len
        out.append((a, b))
    return [(max(0.0, a - pad), min(total, b + pad)) for a, b in out]


_MEMO: dict[str, list[dict]] = {}   # chunk audio hash -> segments (this process)


class WhisperEngine:
    """Thin wrapper over faster-whisper; loads lazily, prefers CUDA float16."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._model = None
        self.device_used = None

    def _load(self):
        if self._model is not None:
            return self._model
        from ..integrations.cuda_env import ensure_cuda_dlls

        ensure_cuda_dlls()
        from faster_whisper import WhisperModel

        dev, ct = self.cfg["device"], self.cfg["compute_type"]
        try:
            self._model = WhisperModel(self.cfg["model"], device=dev, compute_type=ct)
            self.device_used = dev
        except Exception as e:  # noqa: BLE001
            if dev != "cuda":
                raise
            # explicit, reported fallback
            self._model = WhisperModel(self.cfg["model"], device="cpu", compute_type="int8")
            self.device_used = f"cpu (CUDA hatası: {e})"
        return self._model

    def transcribe(self, audio: np.ndarray, sr: int, prompt: str | None = None) -> list[dict]:
        assert sr == 16000, "faster-whisper expects 16 kHz"
        import hashlib
        # the same chunk of sound gives the same words: the 2nd/3rd verification round re-uses ~65 % of the chunks
        # (only the parts around the new cuts change), measured 23.10.2026
        key = hashlib.sha1(np.ascontiguousarray(audio).tobytes() + f"|{self.cfg['model']}|{self.cfg['beam_size']}|"
                           f"{prompt}".encode("utf-8")).hexdigest()
        if key in _MEMO:
            return [dict(s, words=[dict(w) for w in s["words"]]) for s in _MEMO[key]]
        out = self._transcribe_retry(audio, sr, prompt)
        if len(_MEMO) > 6000:
            _MEMO.clear()
        _MEMO[key] = out
        return [dict(s, words=[dict(w) for w in s["words"]]) for s in out]

    def _transcribe_retry(self, audio: np.ndarray, sr: int, prompt: str | None = None) -> list[dict]:
        import time as _t

        for attempt in range(3):
            try:
                return self._transcribe(audio, sr, prompt)
            except RuntimeError as e:
                msg = str(e)
                if "cannot be loaded" not in msg and "cublas" not in msg.lower() and "cudnn" not in msg.lower():
                    raise
                if attempt < 2:
                    _t.sleep(3.0)  # Smart App Control reputation checks can pass on retry
                    continue
                # give up on the GPU, keep working on CPU and say so
                from faster_whisper import WhisperModel

                self._model = WhisperModel(self.cfg["model"], device="cpu", compute_type="int8")
                self.device_used = f"cpu (GPU kitaplığı yüklenemedi: {msg[:160]})"
                return self._transcribe(audio, sr, prompt)
        raise RuntimeError("unreachable")

    def language(self, audio: np.ndarray) -> str:
        """"tr" or "en" for one phrase. The speaker talks Turkish but his wife and the waiters speak English: forcing
        Turkish turned "chili sauce", "mushroom", "crab", "I'm eating" into Turkish-looking nonsense. English is
        chosen only when the model is clearly sure; anything else stays Turkish."""
        langs = self.cfg.get("languages") or [self.cfg.get("language", "tr")]
        if len(langs) < 2 or len(audio) < 16000 * 1.2:
            return langs[0]
        m = self._load()
        if not hasattr(m, "detect_language"):
            return langs[0]
        try:
            _lang, _p, allp = m.detect_language(audio)
        except Exception:  # noqa: BLE001 - older faster-whisper: keep the main language
            return langs[0]
        d = dict(allp or [])
        main, other = langs[0], langs[1]
        if d.get(other, 0.0) >= 0.6 and d.get(other, 0.0) > 2.0 * d.get(main, 0.0):
            return other
        return main

    def _transcribe(self, audio: np.ndarray, sr: int, prompt: str | None = None) -> list[dict]:
        model = self._load()
        lang = self.language(audio)
        segs, _info = model.transcribe(
            audio,
            language=lang,
            beam_size=self.cfg["beam_size"],
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False,
            temperature=[0.0, 0.2, 0.4],
            initial_prompt=prompt if lang == self.cfg.get("language", "tr") else self.cfg.get("prompt_en"),
        )
        out = []
        for s in segs:
            out.append({
                "start": s.start, "end": s.end, "text": s.text, "lang": lang,
                "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
                "words": [{"word": w.word, "start": w.start, "end": w.end, "prob": w.probability} for w in (s.words or [])],
            })
        return out


def vocabulary_prompt(cfg: dict, replacements: dict) -> str | None:
    """A short context sentence for Whisper with the words this channel uses (food, places, names) plus every
    spelling the user corrected before: Whisper then picks "dana yanağı" over "yalan ederken"."""
    words = list(dict.fromkeys([*(cfg.get("vocabulary") or []),
                                *[v for k, v in (replacements or {}).items() if not str(k).startswith("_")]]))
    if not cfg.get("prompt") and not words:
        return None
    return ((cfg.get("prompt") or "") + " " + ", ".join(words[:80]) + ".").strip()


def transcribe_audio(audio: np.ndarray, sr: int, cfg: dict, acfg: dict, dictionary: dict,
                     engine, progress: Callable[[float, str], None] | None = None) -> dict:
    total = len(audio) / sr
    prof = energy_profile(audio, sr, acfg)
    speech = active_regions(prof, acfg)
    chunks = plan_chunks(speech, cfg["chunk_max_s"], cfg["chunk_min_pause_s"], cfg["chunk_pad_s"], total)
    words: list[dict] = []
    dropped: list[dict] = []
    prompt = vocabulary_prompt(cfg, dictionary or {})
    chunks = [c for c in chunks if c[1] - c[0] >= 0.1]
    # (faster-whisper's batched pipeline was measured on 23.10.2026: 11 % faster but 8 % of the words lost - not used)

    def one_by_one():
        for ci, (a, b) in enumerate(chunks):
            if progress:
                progress(ci / max(1, len(chunks)), f"Transkript {ci + 1}/{len(chunks)}")
            seg_audio = audio[int(a * sr):int(b * sr)]
            for s in (engine.transcribe(seg_audio, sr, prompt) if prompt else engine.transcribe(seg_audio, sr)):
                yield ci, s
    for ci, s in one_by_one():
        a = chunks[ci][0]
        text = s["text"].strip()
        if is_hallucination(text, cfg["hallucination_patterns"]) and (
            s["avg_logprob"] < cfg["hallucination_max_avg_logprob"]
            or s["no_speech_prob"] > cfg["hallucination_min_no_speech_prob"]
            or len(text.split()) <= 6
        ):
            dropped.append({"start": a + s["start"], "end": a + s["end"], "text": text, "reason": "hallucination"})
            continue
        for w in s["words"]:
            ww = {"word": w["word"].strip(), "start": round(a + w["start"], 3), "end": round(a + w["end"], 3),
                  "prob": round(float(w["prob"]), 3), "chunk": ci}
            if s.get("lang", "tr") != cfg.get("language", "tr"):
                ww["lang"] = s["lang"]
            if not norm_word(ww["word"]):
                continue
            dur = ww["end"] - ww["start"]
            if ww["prob"] < cfg["uncertain_word_max_prob"] and dur < cfg["uncertain_word_max_dur_s"]:
                if ww["prob"] < cfg["garbage_word_max_prob"]:
                    # cough/breath that Whisper turned into a word: not a word
                    dropped.append({**ww, "reason": "uncertain_short_sound"})
                    continue
                # might be a real short word ("bir", "bu"): keep it (protects the audio), flagged
                ww["uncertain"] = True
            words.append(ww)
    words.sort(key=lambda w: w["start"])
    # chunk padding can make neighbouring chunks overlap: drop a word duplicated across a chunk boundary
    dedup: list[dict] = []
    for w in words:
        if dedup and w["chunk"] != dedup[-1]["chunk"] and w["start"] < dedup[-1]["end"] - 0.05 \
                and norm_word(w["word"]) == norm_word(dedup[-1]["word"]):
            continue
        dedup.append(w)
    words = correct_words(dedup, dictionary)
    if progress:
        progress(1.0, "Transkript tamam")
    return {"duration": total, "chunks": [list(c) for c in chunks], "words": words, "dropped": dropped,
            "device": getattr(engine, "device_used", None),
            "energy": {"floor_db": prof.floor_db, "speech_db": prof.speech_db, "threshold_db": prof.threshold_db}}
