# CallMetric Live ASR — Project Progress

Bu dosya, CallMetric gerçek zamanlı transkripsiyon ve canlı koçluk
projesinde tamamlanan geliştirmeleri kronolojik olarak takip eder.

Her tamamlanan geliştirme görevinin ardından bu dosya güncellenir.

---

## Proje Hedefi

AWS tarafından gönderilen çağrı merkezi seslerini yaklaşık 2 saniyelik
parçalar hâlinde işleyerek:

1. Gerçek zamanlı transkript oluşturmak
2. Görüşmenin niyet ve risklerini analiz etmek
3. Gerektiğinde şirket bilgi tabanında RAG araması yapmak
4. LLM ile müşteri temsilcisine kısa ve kaynaklı öneriler sunmak
5. Bütün süreci ayrıntılı dashboard üzerinden göstermek

Planlanan ana akış:

```text
Audio Chunk
→ Streaming ASR
→ Partial / Stable Transcript
→ Kurallar + SetFit
→ Decision Gate
→ Gerekiyorsa RAG
→ Gerekiyorsa LLM
→ Live Coaching Dashboard
## 18. Tenant-Safe Rolling Audio Buffer

Tarih: 22 Temmuz 2026

Amaç:

Canlı sistemde gelen sıralı ses parçalarının yalnızca son belirli zaman
aralığını bellek içinde güvenli şekilde tutmak.

Eklenenler:

- Varsayılan 20 saniyelik rolling audio buffer oluşturuldu.
- İlk chunk ile tenant, call ve ses formatı buffer'a bağlanır.
- Sonraki chunk'ların aynı tenant ve çağrıya ait olması zorunludur.
- Sample rate, kanal sayısı ve codec değişiklikleri reddedilir.
- Sequence numaralarının kesintisiz ve artan olması zorunludur.
- Tekrarlanan, geriye giden veya eksik sequence numaraları reddedilir.
- Üst üste binen veya geriye giden chunk zamanları reddedilir.
- Rolling window dışında tamamen kalan eski chunk'lar otomatik çıkarılır.
- Son kısa chunk desteklenir.
- Buffer içeriği değiştirilemeyen tuple olarak alınır.
- clear() sonrasında buffer başka tenant ve çağrı için yeniden kullanılabilir.
- Ham audio byte'ları yazdırılmaz veya loglanmaz.

Değişen dosyalar:

```text
app/streaming/rolling_buffer.py
tests/test_rolling_buffer.py
```

## 19. Safe File-Based Audio Streaming Simulator

Tarih: 22 Temmuz 2026

- Yerel ses dosyalarini sirali, tenant ve call bilgilerini koruyan chunk event'leri
  olarak rolling buffer'a aktaran guvenli simulator eklendi.
- Hizli mod beklemeden calisir; gercek zamanli mod injectable sleep ile her
  chunk'in gercek suresini bekler.
- Ham ses verisi icermeyen immutable StreamStep ve JSON-lines CLI eklendi.
- Degisen dosyalar: `app/streaming/simulator.py`,
  `scripts/simulate_audio_stream.py`, `tests/test_streaming_simulator.py`,
  `PROJECT_PROGRESS.md`.
- Testler: focused 36 passed; full 129 passed (1 dependency warning).
- Kalite: Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: Simulator ciktisini gelecekteki streaming ASR katmanina
  baglamak.

## 20. Exact PCM ASR Audio Window

Tarih: 22 Temmuz 2026

- Rolling buffer iceriginden immutable, tenant-aware ve frame-aligned PCM ASR
  penceresi olusturan builder eklendi.
- `pcm_s16le` sesler kronolojik birlestirilir ve ilk kismi kesen pencere siniri
  tam kanal frame'leri korunarak kirpilir; ham PCM guvenli metadata'ya eklenmez.
- Degisen dosyalar: `app/streaming/audio_window.py`,
  `tests/test_audio_window.py`, `PROJECT_PROGRESS.md`.
- Testler: focused ve full kalite kontrolleri tamamlandi.
- Sonraki planli adim: PCM penceresini gelecekteki ASR inference katmanina baglamak.

## 21. In-Memory ASR Window Transcription Adapter

Tarih: 22 Temmuz 2026

- `pcm_s16le` ASR pencerelerini gecici dosya olusturmadan Faster-Whisper'in
  mevcut model yasam dongusuyle transcribe eden injectable adapter eklendi.
- Tenant/call kimlikleri, pencere metadata'si ve clamp edilmis mutlak segment
  zamanlari immutable ve ham ses icermeyen sonuclarda korunur.
- Degisen dosyalar: `app/asr/faster_whisper_engine.py`,
  `app/streaming/window_transcriber.py`, `tests/test_window_transcriber.py`,
  `PROJECT_PROGRESS.md`.
- Testler: focused 23 passed; focused Ruff ve Pyright basarili.
- Sonraki planli adim: Adapter'i gelecekteki rolling streaming orchestration
  katmanina baglamak.

## 22. Deterministic Partial/Stable Transcript Reconciler

Tarih: 22 Temmuz 2026

- Overlapping ASR pencere sonuclarini tenant ve call kapsamini koruyan sirali
  PARTIAL, STABLE ve FINAL transcript event'lerine donusturen reconciler eklendi.
- Normalize edilmis kelime suffix/prefix overlap'i tekrar eden stable metni
  engellerken secilen ozgun ASR yazimini korur; state tamamen resetlenebilir.
- Degisen dosyalar: `app/streaming/transcript_reconciler.py`,
  `tests/test_transcript_reconciler.py`, `PROJECT_PROGRESS.md`.
- Testler: focused 37 passed; focused Ruff ve Pyright basarili.
- Sonraki planli adim: Reconciler'i gelecekteki streaming orchestration katmanina
  baglamak.

## 23. Core File-Based Streaming ASR Pipeline

Tarih: 22 Temmuz 2026

- Tenant ASR ayarlariyla chunk generator, rolling buffer, exact audio window,
  injectable transcriber ve transcript reconciler'i baglayan pipeline eklendi.
- Her chunk icin immutable ve ham ses icermeyen snapshot uretilir; pending partial
  metin dosya sonunda FINAL event olarak CallState'e uygulanir.
- Degisen dosyalar: `app/streaming/pipeline.py`,
  `tests/test_streaming_pipeline.py`, `PROJECT_PROGRESS.md`.
- Testler: focused 42 passed; focused Ruff ve Pyright basarili.
- Sonraki planli adim: Pipeline icin ayri bir kullanici arayuzu gereksinimini
  degerlendirmek.

## 24. Local Streaming ASR CLI

Tarih: 22 Temmuz 2026

- Mevcut Faster-Whisper engine, in-memory window transcriber ve streaming
  pipeline'i CPU/int8 ayarlariyla calistiran yerel dosya CLI'i eklendi.
- Guvenli ayar ve ozet ciktisi, opsiyonel chunk snapshot'lari ve yalnizca acikca
  istendiginde final transcript yazimi desteklenir.
- Degisen dosyalar: `scripts/transcribe_streaming_file.py`,
  `tests/test_transcribe_streaming_file.py`, `PROJECT_PROGRESS.md`.
- Testler: focused 45 passed; focused Ruff ve Pyright basarili.
- Sonraki planli adim: Gercek yerel sesle kontrollu manuel dogrulama.

## 25. PCM Codec Name Canonicalization Fix

Tarih: 22 Temmuz 2026

- PyAV tarafindan uretilen `pcm_s16` codec adi event olusumunda canonical
  `pcm_s16le` degerine normalize edildi; window builder kontrolu korundu.
- Big-endian, unsigned, float ve compressed codec adlari destek kapsamina
  alinmadi.
- Degisen dosyalar: `app/events/models.py`, `app/streaming/audio_window.py`,
  `tests/test_event_models.py`, `tests/test_chunk_generator.py`,
  `tests/test_audio_window.py`, `PROJECT_PROGRESS.md`.
- Testler: focused 37 passed; focused Ruff ve Pyright basarili.
- Sonraki planli adim: Yerel streaming ASR testini tekrar calistirmak.

## 26. Streaming Segment Timestamp Boundary Fix

Tarih: 22 Temmuz 2026

- Kisa ASR pencerelerinde decoder padding nedeniyle pencere disina tasan segment
  zamanlari gercek pencere sinirlarina kirpiliyor.
- Tamamen pencere disindaki segmentler ve metinleri atlanirken non-finite,
  non-numeric ve ters zaman araliklari guvenli metadata hatasiyla reddediliyor.
- Degisen dosyalar: `app/streaming/window_transcriber.py`,
  `tests/test_window_transcriber.py`, `PROJECT_PROGRESS.md`.
- Testler: focused 16 passed; focused Ruff ve Pyright basarili.
- Sonraki planli adim: Yerel streaming ASR testini tekrar calistirmak.

## 27. In-Memory Whisper Sample-Rate Fix

Tarih: 22 Temmuz 2026

- ASR window PCM verisini mono normalized float32 waveform'e donusturen ve kaynak
  hizi farkliysa PyAV ile 16000 Hz'e resample eden tek bir helper eklendi.
- 8000 Hz call-center sesi artik sureyi koruyarak iki kat sample ile Whisper'a
  aktarilir; 16000 Hz girdi gereksiz resample edilmez.
- Degisen dosyalar: `app/streaming/window_transcriber.py`,
  `tests/test_window_transcriber.py`, `PROJECT_PROGRESS.md`.
- Testler: focused 43 passed; focused Ruff ve Pyright basarili.
- Sonraki planli adim: Yerel streaming ASR testini tekrar calistirmak.

## 28. Tenant-Aware Rule-Based Coaching Engine

Tarih: 22 Temmuz 2026

- Tenant label ve izinli action ayarlarina gore STABLE/FINAL transcript event'lerini
  deterministik Unicode-aware kurallarla degerlendiren coaching engine eklendi.
- Unique classification label'lari, en guclu action ve yalnizca template/escalation
  kurallari icin deduplicate edilmis suggestion event'leri uretilir.
- Degisen dosyalar: `app/coaching/__init__.py`, `app/coaching/rule_engine.py`,
  `tests/test_rule_engine.py`, `PROJECT_PROGRESS.md`.
- Testler: focused 48 passed; focused Ruff ve Pyright basarili.
- Sonraki planli adim: Coaching engine'i transcript event akisi ile baglamak.

## 29. In-Memory Coaching Coordinator

Tarih: 22 Temmuz 2026

- Rule-based coaching sonucunu CallState ile baglayan tenant/call-aware coordinator
  eklendi.
- Suggestion content fingerprint deduplication, cooldown ve processing basina
  maksimum suggestion siniri uygulanirken classification sonuclari korunur.
- Degisen dosyalar: `app/coaching/rule_engine.py`,
  `app/coaching/coordinator.py`, `tests/test_coaching_coordinator.py`,
  `PROJECT_PROGRESS.md`.
- Testler: focused 45 passed; focused Ruff ve Pyright basarili.
- Sonraki planli adim: Coordinator'i streaming transcript event akisi ile baglamak.

## 30. Sentetik Canli Kocluk Dashboard Prototipi

- Iki tenant icin izole sentetik Turkce senaryolarla Streamlit canli kocluk paneli eklendi.
- Partial metinler yalnizca ekrani gunceller; stable/final olaylar gercek kural motoru ve coordinator uzerinden islenir.
- Transkript, risk/niyet, oneri ve bastirma, demo gecikme, olay zaman cizelgesi ve mimari durum gorunumleri eklendi.
- Degisen dosyalar: `live_dashboard/app.py`, `live_dashboard/demo_data.py`, `live_dashboard/view_models.py`, `tests/test_live_dashboard_view_models.py`, `PROJECT_PROGRESS.md`.
- Testler: focused 9 passed; full 248 passed (1 dependency warning).
- Kalite: Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: Prototipi sentetik kullanici geri bildirimiyle degerlendirmek.

## 31. Yerel Dosya ve Guvenli Ses Yukleme Modu

- Responsive durum ve mimari kartlari, Turkce aksiyon etiketleri ve baslangic yonlendirmesiyle dashboard yerlesimi iyilestirildi.
- Acik Baslat kapisi ardinda mevcut streaming ASR pipeline ve coaching coordinator kullanan yerel dosya modu eklendi.
- Kendi Sesimle Test bolumu yuklemeyi yalnizca gecici OS dosyasinda isler ve islem sonunda siler; ham ses veya transkript kalici tutulmaz.
- Gercek ASR pencere gecikmeleri, toplam islem suresi, RTF, ilerleme ve istege bagli final transkript indirme gorunumleri eklendi.
- Degisen dosyalar: `app/streaming/pipeline.py`, `live_dashboard/app.py`, `live_dashboard/uploaded_audio.py`, `live_dashboard/view_models.py`, `tests/test_live_dashboard_view_models.py`, `tests/test_streaming_pipeline.py`, `PROJECT_PROGRESS.md`.
- Testler: focused 25 passed; full 255 passed (1 dependency warning).
- Kalite: Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: Yerel modun yalnizca sentetik bir ses dosyasiyla manuel kullanilabilirlik kontrolu.

## 32. Yuklenen Ses Icin Kesin Ilerleme ve ETA

- ASR baslamadan once gercek son kisa parcayi da iceren toplam parca sayisi ve ses suresi belirlenir.
- Parca yuzdesi, zaman araligi, duvar saati, rolling ASR ortalamasi, guvenli ETA, asama ve hata durumu gorunumleri eklendi.
- Tamamlanma ozeti toplam/tamamlanan parca, ses ve islem suresi, ortalama ASR ve RTF degerlerini gosterir.
- Kullaniciya gorunen tenant adlari, ic ID'ler korunarak Demo Telekom ve Demo Yazilim olarak degistirildi.
- Degisen dosyalar: `app/streaming/pipeline.py`, `live_dashboard/app.py`, `live_dashboard/demo_data.py`, `live_dashboard/view_models.py`, `tests/test_live_dashboard_view_models.py`, `tests/test_streaming_pipeline.py`, `PROJECT_PROGRESS.md`.
- Testler: focused 30 passed; full 260 passed (1 dependency warning).
- Kalite: Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: Ilerleme gorunumunu sentetik kisa bir ses dosyasiyla manuel dogrulamak.

## 33. Uc Gorunumlu Dashboard Sunumu

- Dashboard sunumu Temsilci Gorunumu, Teknik Izleme ve Gorusme Sonucu sekmelerine ayrildi.
- Temsilci ekrani kompakt durum/ilerleme, kesin/kismi transkript, intent-risk chip'leri ve zengin kocluk kartlariyla sadeleştirildi.
- Teknik metrikler, ASR cizgisi, pipeline durumlari ve guvenli uyarilar yalnizca Teknik Izleme sekmesine tasindi.
- Tamamlanan gorusmeler icin guvenli metadata, etiket ve oneri ozeti ile istege bagli transkript indirme sunuldu.
- Sidebar Gorusme, Ses Kaynagi ve kapali Model Ayarlari gruplarina ayrildi; geri bildirim yalnizca session_state icinde tutulur.
- Degisen dosyalar: `live_dashboard/app.py`, `live_dashboard/view_models.py`, `tests/test_live_dashboard_view_models.py`, `PROJECT_PROGRESS.md`.
- Testler: focused 26 passed; full 267 passed (1 dependency warning).
- Kalite: Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: Uc gorunumu sentetik bir gorusmeyle manuel kullanilabilirlik testinden gecirmek.

## 34. SetFit Siniflandirma Taksonomisi ve Veri Seti Temeli

- Sekiz etiketli Turkce taksonomi, immutable siniflandirma ornek modeli ve
  guvenli JSONL veri seti yukleyicisi eklendi.
- Duplicate ID, normalize metin tekrari, split sizintisi, bilinmeyen etiket ve
  `no_action` birlikteligi dogrulamalari eklendi.
- Her etiket icin en az alti ornek iceren 48 satirlik, yalnizca sentetik Turkce
  seed veri seti ve metin yazdirmayan guvenli sayim CLI'i eklendi.
- Degisen dosyalar: `app/classification/__init__.py`,
  `app/classification/models.py`, `app/classification/dataset.py`,
  `config/classification_taxonomy.json`,
  `data/synthetic/classification_seed.jsonl`,
  `scripts/validate_classification_dataset.py`,
  `tests/test_classification_models.py`,
  `tests/test_classification_dataset.py`, `PROJECT_PROGRESS.md`.
- Testler: focused 16 passed; full 283 passed (1 dependency warning).
- Kalite: Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: Gelecek ayri bir gorevde SetFit egitim ve degerlendirme
  akisinin tasarlanmasi.

## 35. Genel Multi-Label SetFit Baseline Temeli

- `common_turkish_setfit_v2` icin sabit taksonomi sirali multi-hot encoding,
  one-vs-rest SetFit egitim orchestration'i ve CPU CLI'i eklendi.
- Train split'i fit, validation split'i gelistirme degerlendirmesi icin
  kullanilir; test split'i egitim factory'sine aktarilmaz.
- Threshold ve `no_action` dislayiciligi sonrasi micro/macro ve etiket bazli
  metrikler, exact match, hamming loss ve ortalama inference suresi eklendi.
- Checksum, parametre, split sayimi, zaman ve cozulmus paket surumlerini tutan
  metinsiz metadata ile guvenli JSON degerlendirme raporu eklendi.
- SetFit, scikit-learn ve uyumlu Transformers siniri `uv` ile eklendi; model
  indirilmedi veya egitilmedi.
- Degisen dosyalar: `app/classification/`, `scripts/train_setfit_baseline.py`,
  `scripts/evaluate_setfit_model.py`, `tests/test_classification_encoding.py`,
  `tests/test_classification_evaluation.py`,
  `tests/test_classification_artifacts.py`, `pyproject.toml`, `uv.lock`,
  `PROJECT_PROGRESS.md`.
- Testler: focused 29 passed; full 296 passed (1 dependency warning).
- Kalite: Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: Baseline'i yalnizca acikca onaylanan ayri bir calistirmada
  egitip validation/test raporlarini karsilastirmak.

## 36. Sentetik Veri Genisletme ve Guvenli Olasilik Diagnostikleri

- Genel Turkce siniflandirma seed'i 273 benzersiz sentetik ornege genisletildi;
  split sayilari train 165, validation 54 ve test 54 oldu.
- Her etiket en az train 25, validation 8 ve test 8 ornekte yer alir; konusma
  gruplarinin split'ler arasinda gecisi ve yetersiz label dengesi reddedilir.
- Fiyat sorusu/itirazi, iptal/churn, teknik soru/ariza, sikayet/notr geri bildirim,
  yenileme bilgisi/niyeti ve negation/ASR-benzeri zor karsitliklar eklendi.
- Validation raporlarina metin veya tekil tahmin saklamadan probability
  min/mean/max, threshold gecisleri, TP/FP/FN, bos tahmin ve `no_action`
  threshold/conflict sayimlari eklendi.
- Degisen dosyalar: `app/classification/`, `data/synthetic/classification_seed.jsonl`,
  `scripts/build_classification_seed.py`,
  `scripts/validate_classification_dataset.py`, ilgili classification testleri
  ve `PROJECT_PROGRESS.md`.
- Testler: focused 33 passed; full 300 passed (1 dependency warning); veri seti
  CLI dogrulamasi basarili.
- Kalite: Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: Ayrica onaylanan bir calismada yalnizca validation
  probability diagnostiklerini incelemek.

## 37. Validation-Only Threshold Kalibrasyon Temeli

- Her label icin validation probability'lerinde sinirli ve deterministik threshold
  aramasi eklendi; normal label'lar F1, kritik label'lar recall hedefiyle secilir.
- `cancellation_request`, `churn_risk` ve `complaint` icin recall 0.70 hedefi ve
  hedefe ulasilamazsa recall/precision fallback politikasi uygulandi.
- `no_action` ayri kalibre edilir, mevcut dislayicilik korunur ve business label
  ile `no_action` birlikte esik alti kalan ornek sayisi raporlanir.
- Metin veya tekil tahmin icermeyen before/after metric, checksum, threshold ve
  calibration configuration JSON raporu ile validation-only CLI eklendi.
- Expanded-dataset model kimligi `common_turkish_setfit_v2` olarak duzeltildi;
  rapor model ID'sini artifact metadata'dan aynen korur.
- Degisen dosyalar: `app/classification/calibration.py`,
  `app/classification/artifacts.py`, `scripts/calibrate_setfit_thresholds.py`,
  `tests/test_classification_calibration.py`, `PROJECT_PROGRESS.md`.
- Testler: focused 21 passed; full 307 passed (1 dependency warning).
- Kalite: Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: Egitilmis v2 artifact ile yalnizca validation
  kalibrasyonunu ayrica onaylanan bir calismada calistirmak.

## 38. Versioned Calibrated Threshold Profili

- `common_turkish_setfit_v2` icin schema v1, validation kaynakli ve calibration
  report checksum'larini tasiyan guvenli threshold profili eklendi.
- Profil taxonomy label'larini birebir, threshold araligini, model ID ve
  model/dataset/taxonomy checksum uyumlulugunu dogrular; stale profiller reddedilir.
- Evaluation CLI'e opsiyonel `--threshold-profile` eklendi; verilmezse mevcut
  taxonomy default threshold davranisi aynen korunur.
- Evaluation raporuna metin veya tekil tahmin eklemeden `threshold_source` ve
  `threshold_profile_id` provenance alanlari eklendi.
- `no_action` dislayicilik davranisi degistirilmedi; model inference, validation
  veya test evaluation calistirilmadi.
- Degisen dosyalar: `app/classification/threshold_profiles.py`,
  `app/classification/artifacts.py`, `scripts/evaluate_setfit_model.py`,
  `config/classification_thresholds/common_turkish_setfit_v2.json`, ilgili
  classification testleri ve `PROJECT_PROGRESS.md`.
- Testler: focused 25 passed; full 318 passed (1 dependency warning).
- Kalite: Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: Profil destekli evaluation'i ayrica onaylanan bir
  validation veya test calismasinda kullanmak.

## 39. Lazy Tenant-Aware SetFit Runtime Adapter

- Yerel `common_turkish_setfit_v2` artifact ve calibrated threshold profilini ilk
  istekte dogrulayip yukleyen, artifact seti bazinda tekrar kullanan adapter eklendi.
- Tenant bazli model/profile/taxonomy path override ve ortak v2 default ayarlari
  desteklendi; stale model, taxonomy veya profile checksum'lari reddedilir.
- Runtime sonucu mevcut `ClassificationResultEvent` icinde tenant/call/event
  kimligi, aktif label/score, tum probability/threshold, model/profile ID ve
  inference suresini tasir.
- `no_action` yalnizca model probability'si threshold'u gectiginde uretilir ve
  business label ile birlikte bulunmasi mevcut dislayicilikla engellenir.
- Loglar yalnizca guvenli kimlik, model/profile, aktif label, sure ve hata tipini
  icerir; transcript, token veya embedding loglanmaz ve tahmin saklanmaz.
- Degisen dosyalar: `app/classification/runtime.py`,
  `app/classification/__init__.py`, `app/events/models.py`,
  `tests/test_classification_runtime.py`, `PROJECT_PROGRESS.md`.
- Testler: focused 40 passed; full 327 passed (1 dependency warning); tum model
  yukleme ve inference davranisi mock'ludur.
- Kalite: Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: Adapter'i ayri bir gorevde canli akisa baglamadan once
  sentetik contract-level kullanimla dogrulamak.

## 40. Stable Transcript Runtime Classification Entegrasyonu

- Opsiyonel classification stage, yalnizca cumulative stable transcript gercekten
  degistiginde runtime classifier'i cagiracak sekilde ASR pipeline'a eklendi.
- PARTIAL, bos, degismeyen ve duplicate revision event'leri siniflandirilmaz;
  yeni STABLE ve finalize edilen stable metinler tenant/call kapsamiyla islenir.
- Basarili sonuc mevcut `ClassificationResultEvent` olarak doner ve CallState'te
  yalnizca aktif label, model/profile ID, revision/sequence ve inference suresi
  tutulur; probability, threshold veya transcript kopyasi classification
  metadata'sina yazilmaz.
- Classification hatalari guvenli type/code outcome ve metinsiz log uretir,
  ASR akisini durdurmaz; classifier verilmezse mevcut ASR-only davranis korunur.
- Degisen dosyalar: `app/classification/streaming.py`,
  `app/classification/__init__.py`, `app/calls/models.py`,
  `app/streaming/pipeline.py`, `tests/test_streaming_pipeline.py`,
  `PROJECT_PROGRESS.md`.
- Testler: focused 45 passed; full 332 passed (1 dependency warning);
  classifier tamamen mock'ludur.
- Kalite: Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: Classification outcome'larini ayri bir gorevde coaching
  karar katmanina guvenli contract ile baglamak.

## 41. Stable Classification ve Deterministik Coaching Entegrasyonu

- Opsiyonel coaching coordinator factory, her cagri icin pipeline CallState'ini
  kullanarak yalnizca yeni cumulative stable revision'lari isler; PARTIAL,
  degismeyen ve duplicate revision'lar coaching'e girmez.
- Deterministik rule eslesmeleri ve SetFit aktif label'lari tenant kurallari
  uzerinden birlestirilir; suggestion provenance `rule`, `classification` veya
  `both` olarak korunur ve kritik acik kurallar classification kacirsa da calisir.
- Classification hatasi rule-only coaching'i engellemez; coaching hatasi guvenli
  outcome ve metinsiz log uretir, ASR streaming devam eder.
- CallState yalnizca suggestion ID, action, priority, provenance, revision,
  timestamp ve classification katkisi varsa model/profile ID saklar.
- Degisen dosyalar: `app/events/models.py`, `app/coaching/rule_engine.py`,
  `app/coaching/coordinator.py`, `app/calls/models.py`,
  `app/streaming/pipeline.py`, `tests/test_coaching_coordinator.py`,
  `tests/test_streaming_pipeline.py`, `PROJECT_PROGRESS.md`.
- Testler: focused 78 passed; full 339 passed (1 dependency warning); SetFit
  tamamen mock'ludur.
- Kalite: Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: Coaching metadata'sini ayri bir gorevde sunum katmanina
  guvenli bir view-model ile aktarmak.

## 42. Live Dashboard Classification ve Coaching Sunumu

- Yerel dashboard akisi pipeline'in transient classification/coaching
  outcome'larini dogrudan tuketir; inference veya coaching kurallarini yeniden
  calistirmaz, sentetik demo davranisi korunur.
- Kesinlesen ve partial transcript ayri sunulur; sekiz SetFit etiketi istenen
  Turkce adlarla, coaching kartlari priority/action/provenance/revision ve
  yeni-gosterildi durumuyla temsilci gorunumune aktarilir.
- Suggestion ID bazli tekrar engelleme, sakin bos durum ve classification/coaching
  hata mesajlari eklendi; audio akisi ve mevcut ilerleme/ETA/RTF/ASR metrikleri
  korunur.
- Model/profile/revision/inference ve transient probability degerleri yalnizca
  teknik izleme modelindedir; temsilci gorunumunde ham probability bulunmaz.
- Degisen dosyalar: `live_dashboard/view_models.py`, `live_dashboard/app.py`,
  `tests/test_live_dashboard_view_models.py`, `PROJECT_PROGRESS.md`.
- Testler: focused dashboard 34 passed; full 346 passed (1 dependency warning);
  tum runtime sonuclari sentetik/mock'tur.
- Kalite: Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: Canli UI entegrasyonunu kullanici kabul testiyle
  dogrulamak.

## 43. Live Dashboard SetFit ve Coaching Runtime Wiring

- Uploaded-audio pipeline artik mevcut lazy `RuntimeSetFitClassifier`,
  tenant-specific deterministic `RuleBasedCoachingEngine` ve per-call
  `CoachingCoordinator` factory ile dashboard tarafinda kuruluyor.
- Varsayilan v2 model/profile artifact metadata uyumlulugu model agirliklari
  yuklenmeden kontrol edilir; uyumluysa SetFit, kural veya SetFit varsa coaching
  varsayilan acik gelir.
- Model Ayarlari altina SetFit ve canli coaching kontrolleri eklendi; her iki
  oynatma modu ayni service selection ve pipeline wiring'i kullanir.
- Eksik/uyumsuz artifact durumunda guvenli Turkce mesaj ve rule-only coaching
  korunur; cached classifier Streamlit rerun'larinda yeniden olusturulmaz.
- Teknik izleme SetFit, coaching ve rule engine icin gercek active/disabled/failed
  durumlarini gosterir; yerel artifact veya audio yolu sunuma tasinmaz.
- Degisen dosyalar: `live_dashboard/runtime_wiring.py`,
  `live_dashboard/app.py`, `live_dashboard/view_models.py`,
  `tests/test_live_dashboard_runtime_wiring.py`, `PROJECT_PROGRESS.md`.
- Testler: focused dashboard/integration 58 passed; full 350 passed
  (1 dependency warning); model yukleme mock'ludur.
- Kalite: Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: Yerel uploaded-audio akisini operator kabul testiyle
  dogrulamak.

## 44. Classification-Driven Coaching Suggestion Duzeltmesi

- Yedi business classification label'i icin genel, deterministik Turkce coaching
  template mapping'i eklendi; text rule eslesmese de aktif SetFit label'i
  suggestion uretebilir.
- `cancellation_request` icin guvenli dogrulama/tutundurma onerisi ve yaygin
  `iptal etmek/ettirmek`, `iptal islemini baslatin`, `aboneligimi kapatin`
  bicimlerini kapsayan genel explicit rule eklendi.
- `iptal etmek istemiyorum`, `iptal etmeyecegim` ve `iptal talebim yok`
  olumsuzlamalari rule eslesmesinden dislanir; tenant cancellation rule'u
  eslesirse genel rule tekrar suggestion uretmez.
- Provenance classification/rule/both olarak korunur; coordinator cooldown,
  duplicate suppression, priority ve suggestion limitleri degismemistir.
- Degisen dosyalar: `app/coaching/rule_engine.py`,
  `tests/test_rule_engine.py`, `tests/test_coaching_coordinator.py`,
  `tests/test_live_dashboard_view_models.py`, `PROJECT_PROGRESS.md`.
- Testler: focused coaching/streaming/dashboard 106 passed; full 365 passed
  (1 dependency warning); model yuklenmedi.
- Kalite: Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: Uploaded-audio smoke testini ayni sentetik iptal cumlesiyle
  yeniden dogrulamak.

## 45. Fiyat Bilgisi Kontrast Guard ve Suggestion Label Baglantisi

- Final active-label kararina Turkce fiyat bilgisi/fiyat itirazi contrast guard
  eklendi; net bilgi sorusu ve itiraz kaniti yoksa `price_objection` bastirilir.
- Gercek itiraz kaniti veya bilgi sorusu ile birlikte itiraz bulunan metinlerde
  `price_objection` ve multi-label davranisi korunur; threshold degismemistir.
- Guard yalnizca aktif label listesini filtreler; transient raw probability
  degerleri teknik izleme icin ClassificationResultEvent'te korunur.
- `CoachingSuggestionEvent.label_id` suggestion'in kendi label metadata'sini
  tasir; dashboard artik label'i paralel liste pozisyonundan cikarmadigi icin
  siralama iki kartin metadata'sini caprazlayamaz.
- Degisen dosyalar: `app/classification/postprocessing.py`,
  `app/classification/streaming.py`, `app/events/models.py`,
  `app/coaching/rule_engine.py`, `live_dashboard/view_models.py`,
  `tests/test_classification_postprocessing.py`,
  `tests/test_streaming_pipeline.py`,
  `tests/test_live_dashboard_view_models.py`, `PROJECT_PROGRESS.md`.
- Testler: focused classification/coaching/streaming/dashboard 139 passed;
  full 376 passed (1 dependency warning); model yuklenmedi.
- Kalite: Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: Price-information uploaded-audio smoke testini yeniden
  dogrulamak.

## 46. Uploaded-Audio Guvenli Hata Tanisi ve Dosya Oturumu

- Dashboard pipeline exception'lari stage, error class, chunk sequence,
  transcript revision, servis enable durumlari ve component iceren guvenli
  structured failure metadata'sina donusturulur.
- `logger.exception` yalnizca sanitize edilmis wrapper ve guvenli extra metadata
  ile traceback uretir; transcript, filename, path, audio veya probability loga
  girmez.
- Temsilci gorunumu yalnizca genel hata mesaji, teknik izleme ise stage/error
  code/chunk/component alanlarini gosterir.
- Uploaded file content icin yalnizca session-memory SHA-256 identity kullanilir;
  yeni dosya fresh call state/revision/label/suggestion/cooldown/progress ile
  otomatik baslar, ayni dosya normal rerun'da state'i sifirlamaz.
- Manuel reset uploader generation'i tam bir kez degistirir; sonraki ilk secim
  korunur ve cached Whisper/SetFit kaynaklari temizlenmez.
- Uc partial chunk'li finalization regresyonu final stable transcript ve 3/3
  completion ile dogrulandi.
- Degisen dosyalar: `live_dashboard/uploaded_audio.py`,
  `live_dashboard/view_models.py`, `live_dashboard/app.py`,
  `tests/test_live_dashboard_view_models.py`,
  `tests/test_streaming_pipeline.py`, `PROJECT_PROGRESS.md`.
- Testler: focused dashboard/integration 79 passed; full 380 passed
  (1 dependency warning); model yuklenmedi.
- Kalite: Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: Iki farkli uploaded WAV ile operator smoke testini
  tekrarlamak.

## 47. Uploaded-Audio Stale Run State Duzeltmesi

- File switch sirasinda kontroller fresh `upload_session.execution` kullanirken
  final dashboard render'in stale `_local_state()` kullanmasi kaldirildi; tek
  atomik `LocalExecutionState` hem sidebar hem ana gorunumu besler.
- Upload session `selected_file_identity` ve
  `initialized_run_file_identity` alanlarini ayri tutar; genuine file change
  fresh run'i tam bir kez olusturur, normal rerun state'i yeniden sifirlamaz.
- Fresh run status `idle`, stage `Baslatilmadi`, progress 0%, chunk 0/0,
  Start enabled ve Stop disabled olarak baslar; tamamlanmis, failed veya stopped
  eski state'in transcript/label/suggestion/failure/timing alanlari tasinmaz.
- Automatic switch uploader generation'i degistirmez; explicit reset generation'i
  yalnizca bir kez artirir ve ilk sonraki secim korunur.
- Degisen dosyalar: `live_dashboard/view_models.py`,
  `live_dashboard/app.py`, `tests/test_live_dashboard_view_models.py`,
  `PROJECT_PROGRESS.md`.
- Testler: focused dashboard 46 passed; full 383 passed
  (1 dependency warning); model cache wiring mock'larla korundu.
- Kalite: Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: File A tamamla -> File B sec -> Baslat akisini canli
  Streamlit oturumunda yeniden dogrulamak.

## 48. Kisa Cagri Siniflandirma Geriye Uyumluluk Regresyonlari

- Kisa cagrilarda acik iptal talebi ve coaching, Turkce iptal olumsuzlamasi,
  fiyat bilgisi ile fiyat itirazi ayrimi ve gercek fiyat itirazi davranislari
  uc uca sentetik regresyonlarla koruma altina alindi.
- Kisa cagri coaching onerilerinde `both` provenance ve icerik tabanli
  deduplication davranisinin degismedigi ayrica dogrulandi.
- Degisen dosyalar: `tests/test_streaming_pipeline.py`,
  `tests/test_coaching_coordinator.py`, `PROJECT_PROGRESS.md`.
- Testler: focused streaming/coaching 48 passed; full 388 passed
  (1 dependency warning); model yuklenmedi.
- Kalite: Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: Uzun cagri artimli siniflandirma uygulanirken bu kisa
  cagri regresyonlarini degisikliksiz gecirmek.

## 49. Uzun Cagri Transkript ve Siniflandirma Iyilestirmeleri

- Rolling-window stabil metin uzlastirmasi zaman ortusmesi, noktalama/Turkce
  case normalizasyonu ve sinirli fuzzy kelime ortusmesiyle eski cumleleri
  yeniden eklemez; daha sonra gercekten tekrarlanan konusma korunur.
- Her stabil revizyonda yalnizca yeni delta ve en fazla iki onceki stabil cumle
  siniflandirilir; partial metin, duplicate revizyon ve classification input
  metni guvenli metadata'ya veya loglara girmez.
- Current revision etiketleri ile call-level etiketler ayrildi; call-level
  metadata ilk/son revizyon, rule/classification/both kaynagi ve guvenli model
  kimliklerini saklar, `no_action` business etiketlerle birlikte tutulmaz.
- Coaching yalnizca guncel stabil delta ve guncel classification sonucu ile
  uretilir; deterministic kurallar, cooldown, deduplication, priority ve
  maksimum oneri davranislari korundu.
- Dashboard "Su Anki Etiketler" ve "Gorusmede Tespit Edilenler" alanlarini
  ayri gosterir; teknik gorunum yalnizca metinsiz bounded-context sayaclarini
  gosterir.
- Degisen dosyalar: `app/calls/models.py`, `app/classification/streaming.py`,
  `app/coaching/coordinator.py`, `app/streaming/pipeline.py`,
  `app/streaming/transcript_reconciler.py`, `live_dashboard/app.py`,
  `live_dashboard/view_models.py`, ilgili bes test dosyasi ve
  `PROJECT_PROGRESS.md`.
- Testler: focused 158 passed; full 396 passed (1 dependency warning);
  gercek model yuklenmedi.
- Kalite: Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: Sentetik olmayan uzun WAV smoke testinde transcript ve
  call-level etiket ayrimini operator ekranindan dogrulamak.

## 50. Canonical Call Label Aggregation ve Revision Diagnostics

- Runtime label siniri sekiz taxonomy etiketiyle sinirlandi; bilinen tenant
  alias'lari canonical business etiketlerine cevrilir, `iptal_riski` ve
  `ayrilma_talebi` yalnizca `cancellation_request` olarak saklanir/gosterilir.
- Classification ve deterministic rule kaniti call aggregate icinde
  rule/classification/both provenance ile birlesir; first/latest revision ve
  classification kaynakli model/profile kimlikleri korunur.
- Her stabil classification revizyonu icin yalnizca revision, current canonical
  labels, newly accumulated labels ve guvenli evidence metadata'si tutan
  revision timeline eklendi; text, probability, token, dosya veya path tutulmaz.
- Revision timeline yalnizca Technical Monitoring'de gosterilir; temsilci
  ekraninda canonical `cancellation_request` "Iptal Talebi" olarak gorunur.
- Degisen dosyalar: `app/events/labels.py`, `app/calls/models.py`,
  `app/classification/postprocessing.py`, `app/classification/streaming.py`,
  `app/coaching/rule_engine.py`, `live_dashboard/view_models.py`,
  `live_dashboard/app.py`, ilgili dort test dosyasi ve `PROJECT_PROGRESS.md`.
- Testler: focused 161 passed; full 399 passed (1 dependency warning);
  gercek model yuklenmedi.
- Kalite: Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: Gercek uzun cagri smoke testinde revision timeline
  uzerinden product/technical detection durumunu dogrulamak.

## 51. Dual-View Long-Call Current Intent Detection

- Her yeni stabil delta hem tek basina hem de en fazla iki onceki cumleyi
  iceren bounded context ile siniflandirilir; iki canonical sonuc mevcut
  contrast guard'larindan once birlestirilir.
- Label bazinda delta/bounded_context/both contribution, iki inference'in
  calisma durumu, sureleri ve canonical view etiketleri metinsiz guvenli
  metadata olarak tutulur; raw probability dashboard state'inde tutulmaz.
- Representative ve Technical current labels ayni CallState current revision
  kaynagindan uretilir; rule-derived `cancellation_request` temsilci ekraninda
  "Iptal Talebi" olarak eksiksiz gorunur.
- Representative coaching kartlarindan evidence/fixture/rule ID'leri
  kaldirildi; priority, action, provenance, status, timestamp ve revision
  gorunumu korundu.
- Degisen dosyalar: `app/events/labels.py`, `app/calls/models.py`,
  `app/classification/postprocessing.py`, `app/classification/streaming.py`,
  `live_dashboard/view_models.py`, `live_dashboard/app.py`,
  `tests/test_streaming_pipeline.py`,
  `tests/test_live_dashboard_view_models.py`, `PROJECT_PROGRESS.md`.
- Testler: focused 166 passed; full 404 passed (1 dependency warning);
  gercek model yuklenmedi.
- Kalite: Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: Gercek uzun cagri smoke testinde delta/context
  contribution ile product ve technical recovery'yi dogrulamak.

## 52. Priority-Aware Active Coaching ve Suggestion History

- CallState coaching kartlarini aktif oneriler ve daha once gosterilmis/gecmise
  tasinmis oneriler olarak ayirir; gosterilen tum kartlar history kaydinda
  kaybolmadan korunur.
- Aktif kapasite doluyken admission sirasi priority/severity, current revision
  ve stabil display order ile belirlenir; yeni current HIGH
  cancellation/churn kartlari eski esit veya dusuk rank'li kartlari degistirir.
- Replacement eski karti silmez, history'ye tasir; duplicate fingerprint,
  cooldown ve yalnizca current evidence'tan coaching uretme davranislari
  korunur.
- Representative Anlik Kocluk aktif kartlari once, kompakt Onceki Oneriler
  history'sini sonra gosterir; internal/evidence ID'leri gostermez.
- Technical Monitoring yalnizca revision, canonical label, priority,
  suppression/replacement reason ve history durumunu iceren guvenli karar
  metadata'si gosterir.
- Degisen dosyalar: `app/calls/models.py`, `app/coaching/coordinator.py`,
  `live_dashboard/view_models.py`, `live_dashboard/app.py`,
  `tests/test_coaching_coordinator.py`,
  `tests/test_live_dashboard_view_models.py`, `PROJECT_PROGRESS.md`.
- Testler: focused 170 passed; full 408 passed (1 dependency warning);
  gercek model yuklenmedi.
- Kalite: Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: Gercek uzun cagri smoke testinde final
  cancellation/churn kartlarinin aktif, price/complaint kartlarinin history
  bolumunde oldugunu dogrulamak.

## 53. Representative Active/History View-Model Runtime Fix

- Representative view-model contract'i explicit `active_suggestions` ve
  guvenli bos tuple varsayilanli `suggestion_history` alanlariyla tamamlandi.
- Dashboard builder aktif kartlari yalnizca mevcut aktif state'ten, history
  kartlarini suggestion-history state'inden kurar; renderer eski oturum
  nesnelerinde bos history ile guvenli calisir.
- Price/complaint history ile final cancellation/churn aktif kart ayrimi,
  duplicate olmamasi ve bos/dolu history rendering regresyon testleri eklendi.
- Degisen dosyalar: `live_dashboard/view_models.py`, `live_dashboard/app.py`,
  `tests/test_live_dashboard_view_models.py`,
  `tests/test_live_dashboard_rendering.py`, `PROJECT_PROGRESS.md`.
- Testler: focused 74 passed; full 411 passed (1 dependency warning);
  gercek model yuklenmedi.
- Kalite: Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: Gercek dashboard smoke testinde final aktif ve history
  kartlarinin crash olmadan gorundugunu dogrulamak.

## 54. Coaching Candidate Lifecycle ve Cooldown Siralamasi

- Ayni revizyondaki rule ve classification kaniti canonical label bazinda tek
  adayda birlestirildi; cooldown yalnizca daha once gercekten gosterilen ayni
  aday icin baslatilir.
- Kapasiteden reddedilen adaylar cooldown baslatmaz; yeni HIGH revizyonlar eski
  HIGH kartlari deterministik label sirasi ile history'ye tasiyarak degistirir.
- Revizyon 7 fiyat, 13 sikayet ve 15 iptal/churn sentetik regresyonunda final
  aktif/history ayrimi ve Representative view-model contract'i dogrulandi.
- Degisen dosyalar: `app/calls/models.py`, `app/coaching/coordinator.py`,
  `live_dashboard/view_models.py`, `tests/test_coaching_coordinator.py`,
  `tests/test_live_dashboard_view_models.py`, `PROJECT_PROGRESS.md`.
- Testler: focused 104 passed; full 489 passed (1 dependency warning).
- Kalite: Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: Ayni 7/13/15 akisini gercek dashboard smoke testinde
  yeniden dogrulamak.
