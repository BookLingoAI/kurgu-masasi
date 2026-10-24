# Premiere köprü paneli

Kaynak: [premiere-pro-mcp](https://github.com/ppmcp/premiere-pro-mcp) projesinin `cep-plugin` klasörü (MIT lisansı, bkz. `LICENSE`).

Kurgu Masası için yapılan değişiklikler:
- `CSXS/manifest.xml`: Node.js bayrakları kaldırıldı (panel Node olmadan "ExtendScript dosya modunda" çalışır) ve panel
  Premiere açılınca kendiliğinden başlar (`StartOn` olayları).
- `CSInterface.js`: `evalScript` çağrısındaki hata düzeltildi.

Panel, uygulamanın `kopru\` klasörüne bırakılan komut dosyalarını okur, Premiere'de çalıştırır ve sonucu yine oraya yazar.
Kurulumda `%APPDATA%\Adobe\CEP\extensions\MCPBridgeCEP` içine kopyalanır ve `bridge_dir.txt` ile bağlanır.
