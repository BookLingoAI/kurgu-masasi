"""Expert knowledge for the board members: short checklists from published sources, read locally.

Collected once (22.10.2026) from public pages; nothing is fetched at run time and no user data goes anywhere.
Every rule carries its source and how certain it is, so an agent can tell a standard from a common practice:
  "standart"  - a published standard or the platform's own help page
  "yayin"     - a broadcaster's / platform's style guide (for its own content; adapted here)
  "yaygin"    - widely repeated practice from third-party sources, not verified officially
  "kullanici" - this user's own decisions (CLAUDE.md) - these override everything else
"""
from __future__ import annotations

SOURCES = {
    "yt_chapters": "https://support.google.com/youtube/answer/9884579",
    "ebu_r128": "https://tech.ebu.ch/docs/r/r128-2014.pdf",
    "ebu_r95": "https://tech.ebu.ch/publications/r095",
    "netflix_ttsg": "https://backlothelp.netflix.com/hc/en-us/articles/215274938",
    "material_motion": "https://m1.material.io/motion/duration-easing.html",
    "rule30": "https://en.wikipedia.org/wiki/30-degree_rule",
    "jump_cut": "https://en.wikipedia.org/wiki/Jump_cut",
    "yt_loudness_3p": "https://youlean.co/?p=30813",
    "hook_3p": "https://subscribr.ai/p/youtube-analytics-improve-video-hooks-intros",
    "humanizer": "https://github.com/blader/humanizer (MIT) / https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing",
}

RULES = {
    "hikaye": [
        ("Bölüm zaman damgaları: ilk 00:00, en az 3 bölüm, her bölüm en az 10 sn.", "yt_chapters", "standart"),
        ("İlk 10-30 sn'de konu/merak açıkça verilmeli; izleyici kaybı en çok açılışta olur.", "hook_3p", "yaygin"),
        ("Gerçek izlenme verisi yoksa izleyici kaybı tahmini HİPOTEZDİR; kesin sonuç gibi söyleme.", None, "kullanici"),
        ("Anlamlı duraklama ve önemli açıklama sırf hız için kesilmez; söylenmeyen bilgi eklenmez.", None, "kullanici"),
        ("Kötü kaba kurgunun üstüne efekt yığılmaz: önce kesim ve akış.", None, "kullanici"),
        ("Başlık/açıklama yapay zekâ gibi durmasın: 'X değil Y' karşıtlığı, uzun tire, 'keşfedin/yolculuk/eşsiz' gibi "
         "kalıp kelimeler, zorlama üçlüler ve tek satırlık dramatik kapanış yok.", "humanizer", "yaygin"),
    ],
    "grafik": [
        ("Ekran yazısı satırı en fazla ~42 karakter.", "netflix_ttsg", "yayin"),
        ("Okuma hızı yetişkin için ~17 karakter/sn; daha hızlı yazı okunmaz (Türkçe için ayrı değer doğrulanmadı).", "netflix_ttsg", "yayin"),
        ("Grafik güvenli alanı: her kenardan %5 içeride; hareket güvenli alanı %3,5.", "ebu_r95", "standart"),
        ("Ekrana yalnız konuşmacının söylediği kelimeler yazılır; altyazıyla aynı metni ikinci kez yazma.", None, "kullanici"),
        ("Emoji önerme: kullanıcı kaldırttı.", None, "kullanici"),
        ("Ekran yazısında abartı ve satış dili yok ('İNANILMAZ', 'ŞOK'); konuşmacının sözü sade yazılır.", "humanizer", "yaygin"),
    ],
    "animasyon": [
        ("Arayüz hareketleri ~200-400 ms; girişler ~225 ms, çıkışlar ~195 ms, 400 ms üstü ağır hisseder "
         "(arayüz kuralı; video yazısına uyarlanır).", "material_motion", "yayin"),
        ("Giren öğe yavaşlayarak durur (ease-out), çıkan hızlanarak gider (ease-in).", "material_motion", "yayin"),
        ("Uzun mesafe/büyük öğe daha uzun süre alır; tek süre her harekete uymaz.", "material_motion", "yayin"),
        ("Kullanıcı yazıların donuk durmasını sevmiyor; düz masa başı çekimde kamera kart olup kenara kayabilir.", None, "kullanici"),
    ],
    "gecis": [
        ("Aynı konuya kesimde kamera açısı ~30° ya da çekim ölçeği belirgin değişmezse kesim 'zıplar' (jump cut).", "rule30", "yayin"),
        ("Vlog ve konuşma videolarında jump cut bilinçli bir üsluptur; her zıplamayı gizlemek gerekmez.", "jump_cut", "yaygin"),
        ("Ölçek değişimi (punch-in) zıplamayı yumuşatır; aynı ölçekte art arda kesim göze batar.", "rule30", "yaygin"),
        ("Programda olmayan bir Premiere özelliğini varmış gibi önerme; plan edit_plan.json üzerinden kurulur.", None, "kullanici"),
        ("Hep aynı geçiş olmasın: zoom / whip / flash sırayla.", None, "kullanici"),
    ],
    "ses": [
        ("Yayın standardı: program -23 LUFS ±0,5 LU, gerçek tepe ≤ -1 dBTP; ses aralığına (LRA) üst sınır yok.", "ebu_r128", "standart"),
        ("YouTube yüksek videoların sesini yaklaşık -14 LUFS'e indirir, sessizleri yükseltmez (resmi değil).", "yt_loudness_3p", "yaygin"),
        ("Bu kanalın hedefi -14 LUFS / tepe ≤ -1 dBTP; ±1 LU içindeyse seviye önerisi yapma.", None, "kullanici"),
        ("Abone/beğeni/yorum sesleri SABİT; nefes tanıyıcı ve güçlü RNNoise önerme.", None, "kullanici"),
    ],
    "renk": [
        ("Renk değişikliği yalnız ölçülmüş farkla önerilir (parlaklık > 8 ya da renk farkı ΔE > 5).", None, "kullanici"),
        ("Patlamış ışık: görüntünün %3'ünden fazlası 250 üstündeyse.", None, "kullanici"),
        ("S-eğrisi, sert kontrast ve cilt rötuşu önerme (rötuş CapCut'ta yapılıyor).", None, "kullanici"),
        ("Denetçi olarak: kanıtı zayıf öneriye somut gerekçeyle itiraz et; gerçek sorun yoksa itiraz uydurma.", None, "kullanici"),
    ],
}
LABEL = {"standart": "STANDART", "yayin": "YAYIN KURALI", "yaygin": "YAYGIN UYGULAMA (doğrulanmamış)",
         "kullanici": "KULLANICININ KURALI"}


def checklist(agent: str) -> str:
    """The agent's checklist as prompt text, user rules first."""
    rules = sorted(RULES.get(agent, []), key=lambda r: r[2] != "kullanici")
    return "\n".join(f"- [{LABEL[k]}] {text}" for text, _src, k in rules)


def sources(agent: str) -> list[str]:
    return sorted({SOURCES[s] for _t, s, _k in RULES.get(agent, []) if s})
