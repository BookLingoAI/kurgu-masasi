# Üçüncü taraf parçalar ve lisansları

Kurgu Masası'nın kendi kodu MIT lisanslıdır. Depoda bulunan ya da kurulumda indirilen başka parçalar:

## Depoda bulunanlar

| Parça | Konum | Lisans |
|---|---|---|
| Premiere köprü paneli ([premiere-pro-mcp](https://github.com/ppmcp/premiere-pro-mcp) `cep-plugin`, değiştirildi) | `kurulum_kaynak/premiere-kopru/` | MIT (`LICENSE` orada) |
| Yazı tipleri: Anton, Bebas Neue, IBM Plex Mono, Instrument Serif, Inter, Montserrat, Oswald, Poppins (Google Fonts) | `assets/fonts/` | SIL Open Font License 1.1 (`assets/fonts/lisanslar/`) |
| [Lucide](https://lucide.dev) simgeleri (51 adet) | `assets/icons/` | ISC (`assets/icons/LICENSE-lucide.txt`) |
| RNNoise modeli `cb.rnnn` ([GregorR/rnnoise-models](https://github.com/GregorR/rnnoise-models)) | `assets/rnnoise/` | telif kapsamında değil (depo README'si) |
| MediaPipe yüz/insan modelleri (BlazeFace, Selfie Segmenter, Face Landmarker) | `modeller/` | Apache-2.0 (Google) |
| HyperFrames katalog blokları ([heygen-com/hyperframes](https://github.com/heygen-com/hyperframes) registry) | `araclar/hyperframes/katalog/` | Apache-2.0 |

## Kurulumda indirilenler (depoda yok)

| Parça | Lisans |
|---|---|
| Python 3.12 | PSF |
| FFmpeg (gyan.dev derlemesi) | GPL-3.0 |
| faster-whisper / CTranslate2, Whisper modelleri | MIT |
| OpenCV, NumPy, SciPy, Pillow, fastembed, onnxruntime, MediaPipe | Apache-2.0 / BSD / MIT / HPND |
| LM Studio | LM Studio kullanım koşulları (ücretsiz) |
| Qwen3 dil modeli | Apache-2.0 |
| HyperFrames (isteğe bağlı, `npm ci`) | Apache-2.0 |
| GSAP (isteğe bağlı, HyperFrames blokları için) | GSAP "Standard No Charge" lisansı |
| Demucs + PyTorch (isteğe bağlı müzik temizleyici) | MIT / BSD |
| Piper `tr_TR-dfki-medium` sesi, Coqui XTTS-v2 (isteğe bağlı ajan sesleri) | CC BY-NC-SA 4.0 / Coqui Public Model License — **yalnız ticari olmayan kullanım** |

Uygulama içinden indirilebilen ses efektleri (Mixkit Free License, Kenney CC0) depoya konmadı; indirilen her dosyanın
kaynağı ve lisansı `paketler\katalog.json` içine yazılır.
