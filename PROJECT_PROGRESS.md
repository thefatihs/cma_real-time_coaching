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
