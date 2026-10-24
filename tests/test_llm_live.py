"""Live check of the local language model (LM Studio). Runs only on the user's PC (KURGU_GPU_TEST=1, set by the
"tests" job) while LM Studio's server is up; prints load time and speed so slow/silent models can be diagnosed
(every call is also logged to logs/llm.log)."""
import json
import os
import time

import pytest

from kurgu.integrations import llm as L

SENTS = ["Dostlarım merhaba, bugün sınırsız Japon hot pot restoranındayız.", "Kişi başı 35 dolar ödüyoruz.",
         "Etleri kendimiz pişiriyoruz, masada iki tencere var.", "Bu sos çok acı ama ben seviyorum.",
         "Hatun sosları hazırladı, bakalım nasıl olmuş.", "Videoyu beğenmeyi unutmayın.",
         "Sushi'ler de geldi, şöyle göstereyim size.", "Kuzu eti inanılmaz yumuşak.",
         "Türkiye'de olsaydı kaç para öderdik sizce?", "Yorumlara tahmininizi yazın.",
         "Garson geldi, daha getireyim mi diyor.", "Toplamda 167 dolar verdik."] * 7


@pytest.mark.skipif(os.environ.get("KURGU_GPU_TEST") != "1", reason="yalnızca kullanıcının bilgisayarında")
def test_language_model_answers_in_time():
    if not L.server_up():
        pytest.skip("LM Studio sunucusu kapalı")
    info = L.models_info()
    print("\nLLM modeller:", json.dumps([{k: m.get(k) for k in ("id", "state", "arch", "quantization", "max_context_length",
                                                               "loaded_context_length", "type")} for m in info], ensure_ascii=False))
    llm = L.LocalLLM({"prefer": "qwen", "no_think": True, "timeout_s": 600})
    llm.key = L.model_key(llm.cfg)
    print("LLM seçilen:", llm.key)
    t = llm.warm(900)
    print(f"LLM yükleme/ilk cevap: {t:.1f} sn")
    lines = "\n".join(f"[{i}] {s}" for i, s in enumerate(SENTS))
    schema = {"type": "object", "properties": {"picks": {"type": "array", "items": {"type": "object", "properties": {
        "i": {"type": "integer"}, "why": {"type": "string"}}, "required": ["i", "why"], "additionalProperties": False}}},
        "required": ["picks"], "additionalProperties": False}
    t0 = time.time()
    out = llm.chat_json("Sen bir YouTube video editörüsün.", f"{lines}\n\nEkrana yazılmaya değer en iyi 5 cümlenin numarasını "
                        "ve 2-3 kelimelik nedenini ver. Sadece JSON.", schema, max_tokens=300, temperature=0.2)
    dt = time.time() - t0
    print(f"LLM JSON cevap: {dt:.1f} sn, hız {L.STATS.get('tok_s')} token/sn, cevap: {json.dumps(out, ensure_ascii=False)[:300]}")
    assert isinstance(out.get("picks"), list)
