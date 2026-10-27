# Kurgu Masası 🎬

**Telefonla çektiğin konuşmalı videoları kendi bilgisayarında, yapay zekâyla otomatik kurgulayan Windows uygulaması.**
Videoyu seçersin; uygulama boşlukları ve "ııı/eee"leri keser, önemli anlara yazılar koyar, yüze yakınlaştırır,
geçiş ve ses efektleri ekler, sesi temizler, renk düzeltir ve YouTube'a hazır bir video çıkarır. Yanında Shorts videosu,
kapak için en iyi kareler, başlık ve açıklama da hazırlar.

**Her şey senin bilgisayarında çalışır.** Videoların, sesin ve yazıların hiçbir buluta gönderilmez.

> English summary at the bottom.

---

## Ne yapar?

| Adım | Ne olur | Kullanılan araç |
|---|---|---|
| 1. Dinleme | Mekândaki müziği konuşmadan ayırır, konuşmayı kelime kelime ve zamanıyla yazıya döker | Demucs (isteğe bağlı), Whisper (faster-whisper) |
| 2. Kesim | Duraklamaları, dolguları (ııı, eee, hmm), öksürükleri, tekrar edilen cümleleri keser; kesilmiş sesi yeniden dinleyip bir şey kaybolmamış mı kontrol eder | kendi kuralları + Whisper |
| 3. Yönetmenlik | Fiyat, yorum gibi anlara ekran yazısı, bölüm kartı ve liste koyar; yüzü ya da gösterilen yemeği bulup yakınlaştırır; konu değişiminde geçiş ve ses efekti; yemek/mekân görüntülerinden müzikli intro | yerel dil modeli (LM Studio), OpenCV |
| 4. Video | Renk düzeltme, ses temizliği ve -14 LUFS (YouTube) ses seviyesi, ekran kartıyla hızlı kodlama | FFmpeg (NVENC) |
| 5. Paket | Shorts (dikey, yüz takipli, kelime kelime altyazılı), kapak için 6 temiz kare, 3 başlık + açıklama + bölümler | yerel dil modeli |
| 6. Premiere (isteğe bağlı) | Aynı kurguyu Adobe Premiere Pro'da düzenlenebilir bir proje olarak kurar | Premiere Pro + köprü paneli |

Ekrana **konuşmada söylenmeyen hiçbir bilgi yazılmaz**: her yazı, söylenen cümleyle karşılaştırılır.

## Neye ihtiyacın var?

| Gerekli | Ne için | Ücret |
|---|---|---|
| **Windows 10 / 11 (64 bit)** | uygulama Windows için yazıldı | — |
| **NVIDIA ekran kartı** (önerilir, 8 GB+) | Whisper ve video kodlama hızlı çalışır; yoksa işlemciyle, yavaş çalışır | — |
| **16 GB RAM, ~30 GB boş disk** | modeller ve videolar | — |
| **LM Studio** + bir **Qwen** dil modeli | yazılar, Shorts seçimi, başlık/açıklama | ücretsiz |
| Python 3.12, FFmpeg | uygulamanın kendisi | ücretsiz (kurulum kendisi kurar) |
| **Adobe Premiere Pro** | *isteğe bağlı*: kurguyu Premiere'de açıp elle düzenlemek için | ücretli |

Premiere olmadan da tam video çıkar; uygulama kendi kurgu motoruyla çalışır.

## Kurulum (tek tık)

1. Bu sayfada **Code → Download ZIP** ile indir (ya da `git clone`) ve bir klasöre çıkar.
2. **`KUR.bat`** dosyasına çift tıkla.
3. Açılan pencerede **Kur**'a bas. Kurulum eksikleri bulur ve ücretsiz olanları indirir:
   Python 3.12, FFmpeg, yapay zekâ paketleri, Whisper modeli, LM Studio ve dil modeli (Qwen3 8B, ~5 GB).
   Premiere Pro kuruluysa köprü panelini de kurar. Masaüstüne **Kurgu Masası** kısayolu koyar.
4. **LM Studio'yu aç**, solda *Developer* sekmesinde sunucuyu başlat (adres `http://127.0.0.1:1234`).
   Model ilk istekte kendiliğinden yüklenir.
5. Masaüstündeki **Kurgu Masası**'nı aç.

> Windows "Bilinmeyen yayıncı" uyarısı verirse *Ek bilgi → Yine de çalıştır* de. Kurulumun yaptığı her şey
> `kurulum_kaynak\kurulum.ps1` dosyasında açıkça okunabilir.

### Daha iyi yazılar için daha büyük model

Kurulum, çoğu ekran kartına sığan **Qwen3 8B**'yi indirir. Ekran kartın büyükse LM Studio'dan daha büyük bir Qwen
modeli (ör. Qwen3.8 27B) indirip `design_config.json` içinde `"llm": {"model": "..."}` alanına adını yazabilirsin.
Ayarlanan model bilgisayarda yoksa uygulama kurulu Qwen modelini kendisi seçer. Model ekran kartına sığmazsa çalışır
ama yavaşlar; uygulama Whisper ve video adımlarında modeli ekran kartından kendisi boşaltır.

## Kullanım

1. **Bilgisayardan seç** ile videoları seç (kopyalanmaz, oldukları yerden okunur). "Hangisi ne?" adımında konuşma kaydı /
   ek görüntü olarak işaretle.
2. Proje sayfasında **Hepsini tek seferde**'ye bas: kesim → video → kapak kareleri → Shorts → başlık/açıklama.
3. Bitince her şey **`Videolarım\<proje adı>\Yüklenecekler`** klasöründedir. O videonun tüm dosyaları
   `Videolarım\<proje adı>` içindedir; temizlik için o klasörü silmen yeterli (asıl kayıtlarına dokunulmaz).

### Yatay mı, dikey mi?

"Yeni video" kartında üç seçenek var:
- **Yatay video edit** (varsayılan): yukarıdaki 16:9 YouTube kurgusu; yanında Shorts, kapak kareleri, başlık ve açıklama.
- **Yataydan dikey video çıkar**: yatay çekimden tek bir dikey (1080x1920) video yapar; yüz takipli kırpma, kelime
  kelime altyazı, -14 LUFS.
- **Dikey video yükle**: dikey çektiğin videoyu dikey olarak kurgular.

Dikey videolar **en fazla 3 dakika** olur, çünkü YouTube 3 dakikadan uzun dikey videoları Shorts saymaz. Kesilmiş
video 3 dakikadan kısaysa tamamı kullanılır; uzunsa en güçlü anlardan (kanca, giriş, gelişme, sonuç) yaklaşık 2-3
dakikalık bir kurgu çıkar. Dosya `Yüklenecekler\1 - Dikey video (Shorts).mp4` olur.

### Kurgu motorları

Proje sayfasının üstünden seçilir:
- **Kurgu Masası** (varsayılan): kendi motoru — renk düzeltme, kamera oyunları, abone animasyonu, altyazı.
- **Karma**: kesim ve kamera Kurgu Masası'ndan, başlık/fiyat/liste yazıları [HyperFrames](https://github.com/heygen-com/hyperframes) bloklarından.
- **HyperFrames**: videoyu baştan sona HyperFrames kurar ve çizer (daha yavaş; renk düzeltme ve altyazı yok).

HyperFrames isteğe bağlıdır ve [Node.js](https://nodejs.org) ister:
```bat
cd araclar\hyperframes
npm ci
```

## Sık karşılaşılanlar

| Sorun | Çözüm |
|---|---|
| "Dil modeline ulaşılamadı" | LM Studio açık mı, *Developer* sekmesinde sunucu çalışıyor mu? |
| Mekân müziği konuşmanın altında kalıyor | Uygulamadaki **Müzik temizleyiciyi kur** düğmesi (~3 GB) |
| "Uygulama Denetimi ilkesi bu dosyayı engelledi" | Windows Akıllı Uygulama Denetimi bazı kitaplıkları ara sıra engelliyor; uygulama yeniden dener, olmazsa işlemciyle devam eder |
| Premiere açılmıyor | Premiere'in Creative Cloud'dan kurulmuş, güncel ve sağlam olduğundan emin ol; olmasa da video çıkar |

## Klasör düzeni

```
kurgu/            uygulamanın kodu (Python)
  analysis/       ses, transkript, metin    editing/   kesim, yönetmen, yazılar, Shorts
  motion/         çizim, yazı animasyonları audio/     ses temizliği, efektler
  vision/         yüz, renk
  integrations/   FFmpeg, LM Studio, Premiere, HyperFrames     web/  arayüz
tests/            otomatik testler
kurulum_kaynak/   kurulum sihirbazı, paket listeleri, Premiere köprü paneli
design_config.json  ayarlanabilir tüm sayılar
Videolarım/       (kurulumdan sonra) her video kendi klasöründe
```

## Geliştiriciler için

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m kurgu            :: arayüz: http://localhost:8770
.venv\Scripts\python.exe -m pytest tests     :: testler
```

## Lisans

Kurgu Masası'nın kodu **MIT** lisanslıdır (`LICENSE`). Kullanılan üçüncü taraf parçaların lisansları `THIRD_PARTY.md` içinde.

---

## English summary

**Kurgu Masası** ("editing desk") is a Windows app that automatically edits talking-head and vlog videos on your own
computer: it cuts pauses, fillers and retakes (Whisper), places on-screen titles only from what was actually said
(local LLM via LM Studio), punches in on faces, adds transitions and sound effects, cleans and masters the audio
(-14 LUFS), colour-corrects and encodes with NVENC. It also makes a Shorts cut, thumbnail frames and YouTube titles,
can build an Adobe Premiere Pro project of the edit (optional). On the upload screen you can pick a horizontal edit, a vertical cut from horizontal footage, or a
vertical upload; vertical videos are capped at 3 minutes (YouTube Shorts limit). Nothing is uploaded anywhere. Install: download, double-click `KUR.bat`, start the LM Studio server.
The UI is in Turkish. License: MIT.
