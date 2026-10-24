"""Real Whisper on the real GPU (Windows only). Speech is synthesized with Windows SAPI.

Runs only when KURGU_GPU_TEST=1 (the 'tests' job sets it on Windows).
"""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from kurgu.config import Config

CFG = Config.load().design
RUN = os.environ.get("KURGU_GPU_TEST") == "1" and os.name == "nt"

PS = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$v = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like 'tr*' } | Select-Object -First 1
if ($v) { $s.SelectVoice($v.VoiceInfo.Name) }
$s.Rate = 0
$s.SetOutputToWaveFile('{OUT}')
$p = New-Object System.Speech.Synthesis.PromptBuilder
{BODY}
$s.Speak($p)
$s.Dispose()
if ($v) { 'VOICE=' + $v.VoiceInfo.Name } else { 'VOICE=none' }
"""


def synth(parts, out):
    body = []
    for kind, val in parts:
        if kind == "say":
            body.append(f"$p.AppendText('{val}')")
        else:
            body.append(f"$p.AppendBreak([TimeSpan]::FromMilliseconds({int(val * 1000)}))")
    script = PS.replace("{OUT}", str(out)).replace("{BODY}", "\n".join(body))
    r = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True, timeout=120)
    return r.stdout.strip()


@unittest.skipUnless(RUN, "GPU testi yalnızca Windows'ta KURGU_GPU_TEST=1 ile")
class TestWhisperGPU(unittest.TestCase):
    def test_chunked_transcript_and_cuts(self):
        from kurgu.analysis.audio import active_regions, energy_profile
        from kurgu.analysis.transcript import WhisperEngine, transcribe_audio
        from kurgu.editing.cuts import compute_cuts
        from kurgu.integrations import ffmpeg as ff

        tmp = Path(tempfile.mkdtemp())
        wav = tmp / "s.wav"
        voice = synth([("say", "Bugün size yeni bir"), ("pause", 0.8),
                       ("say", "Bugün size yeni bir model göstereceğim."), ("pause", 6.0),
                       ("say", "Bu model çok hızlı çalışıyor.")], wav)
        print(voice)
        audio = ff.load_audio(wav, 16000)
        eng = WhisperEngine(CFG["transcript"])
        t = transcribe_audio(audio, 16000, CFG["transcript"], CFG["audio_analysis"], {}, eng)
        print("device:", t["device"])
        print("words:", " ".join(w["word"] for w in t["words"]))
        self.assertEqual(t["device"], "cuda", t["device"])
        self.assertGreaterEqual(len(t["words"]), 8)
        prof = energy_profile(audio, 16000, CFG["audio_analysis"])
        res = compute_cuts(t["words"], active_regions(prof, CFG["audio_analysis"]), len(audio) / 16000, CFG["cuts"])
        dead = set(res["removed_word_idx"])
        kept = " ".join(w["word"] for i, w in enumerate(t["words"]) if i not in dead)
        print("kept:", kept)
        print("rules:", sorted({r["rule"] for r in res["removals"]}))
        if voice != "VOICE=none":
            # Turkish voice available: the abandoned first take must be cut
            self.assertEqual(kept.lower().count("bugün"), 1, kept)
        self.assertIn("long_wait", {r["rule"] for r in res["removals"]})


if __name__ == "__main__":
    unittest.main()
