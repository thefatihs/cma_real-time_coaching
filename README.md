# CallMetric Live ASR

CallMetric Live ASR; tenant kapsamlı konuşma çözümleme, sınıflandırma, temsilci
koçluğu ve belge tabanlı RAG bileşenlerini birleştiren Python 3.12 projesidir.
Geliştirici arayüzü Streamlit dashboard, temel servis arayüzü FastAPI’dir.

## Güncel durum

Depoda Faster-Whisper tabanlı ASR, tenant-aware SetFit sınıflandırma,
deterministik koçluk, Streamlit temsilci görünümü ve **Bilgi Tabanı** sekmesi
bulunur. PDF/TXT/Markdown ingestion; PostgreSQL 16/pgvector registry, job,
embedding ve retrieval; CPU üzerinde 384 boyutlu normalize MiniLM; HTTPS
OpenAI-compatible vLLM; result gate ve güvenli citation akışı uygulanmıştır.

Gerçek Windows dashboard RAG/vLLM doğrulaması tamamlanmış ve controller tam
olarak `E2E_OK` üretmiştir. Bu sonuç üretim dağıtımı veya formal toplam çalışma
süresi garantisi değildir. Ayrıntılı kanıt için
[Windows dashboard E2E runbook](docs/runbooks/windows_dashboard_rag_vllm_e2e.md)
kullanılmalıdır.

## Mimari özet

```text
Yerel ses → ASR → SetFit → koçluk karar kapısı
                              ↓
Belge → güvenli hazırlama → PostgreSQL/pgvector retrieval
                              ↓
                 MiniLM → HTTPS vLLM → result gate
                              ↓
                       güvenli citation
```

Her çağrı olayı `tenant_id` ve `call_id` kapsamını taşır. Belge işlemleri ayrıca
sunucu tarafından doğrulanan `knowledge_base_id` ile sınırlandırılır. Bu kapsam
değerleri dashboard veya upload alanından kabul edilmez.

## Kurulum ve geliştirme

Python 3.12 ve `uv` gereklidir:

```shell
uv sync --locked
uv run uvicorn app.main:app --reload
```

Dashboard:

```shell
uv run streamlit run live_dashboard/app.py
```

Gerçek model veya servis çalıştırmadan önce ilgili runbook’u izleyin. Normal
test akışı model indirmez ve dış servise bağlanmaz.

## Dashboard modları

- **Sentetik Demo** deterministik dashboard davranışını gösterir; PostgreSQL
  retrieval veya vLLM çağrısını tetiklemez.
- **Yerel Ses Dosyası**, UI üzerinden ASR → SetFit → RAG/vLLM koçluk zincirini
  çalıştıran desteklenen manuel yoldur. Uyumlu artifact’ler ve hazır dış
  servisler gerektirir.
- **Bilgi Tabanı**, çağrı başlamadan tenant kapsamlı PDF/TXT/Markdown yükleme,
  bounded ilerleme, listeleme ve onaylı tam-kapsam silme sağlar.

Bir belgenin `READY` olması retrieval’da kullanıldığını kanıtlamaz. Kaynaklar
yalnız başarılı ve non-empty retrieval bağlamı kullanan koçlukta gösterilir.

## Güvenli belge ingestion

- Tür, boyut, PDF sayfa sayısı ve çıkarılan karakter sayısı bounded’dır.
- Şifreli/bozuk PDF, path semantiği taşıyan ad, geçersiz UTF-8 ve boş/aşırı
  içerik fail-closed reddedilir.
- Kaynak byte’ları yalnız bounded işleme süresince bellektedir; başarı, hata,
  iptal veya close sonrasında bırakılır. Kalıcı kaynak nesnesi oluşmaz.
- PostgreSQL güvenli registry alanları, job durumu, sıralı chunk metadata’sı ve
  embedding’leri saklar. Yeni belgelerde `storage_object_key` `NULL` kalır.
- SHA-256 idempotency tenant + knowledge-base kapsamındadır. Silme yalnız exact
  scoped belge/job/vector kayıtlarını etkiler.
- Citation UI iç kimlik veya chunk metni göstermez; güvenli dosya adı, tür ve
  varsa güvenilir sayfa metadata’sıyla sınırlıdır.

Otoritatif ayrıntılar:
[PostgreSQL RAG smoke runbook](docs/runbooks/postgres_rag_smoke.md).

## Yapılandırma ve güvenlik sınırları

Değerleri Git’e yazılmadan kullanılan environment aileleri:

- `CALLMETRIC_DASHBOARD_SMOKE_TENANT_OVERRIDE_PATH`;
- `CALLMETRIC_DASHBOARD_RAG_*` ve `CALLMETRIC_DASHBOARD_DOCUMENT_*`;
- `CALLMETRIC_POSTGRES_*`, `CALLMETRIC_POSTGRES_MIGRATION_*` ve
  `CALLMETRIC_POSTGRES_TLS_SERVICE_*`;
- `CALLMETRIC_VLLM_*`.

Provider/policy JSON dosyaları strict, server-owned ve secret-free olmalıdır.
DSN, token, sertifika içeriği, özel mutlak yol, müşteri verisi ve model cache’i
Git’e, UI’a veya session state’e eklenmez. Exact sözleşmeler runbook’lardadır.

## Kalite kontrolleri

```shell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv lock --check
uv run python scripts/check_conflict_markers.py
git diff --check
```

Gerçek PostgreSQL ve GPU/vLLM doğrulamaları opt-in’dir.

## Güvenli operasyon

Özet başlatma sırası: HTTPS vLLM READY → doğrulanmış, yalnız loopback’e bağlanan
owned SSH tunnel → TLS PostgreSQL/pgvector → private handoff ile dashboard
environment → Streamlit. Tunnel vLLM’i internete açmamalıdır. Özet kapatma
sırası: dashboard Reset ve Streamlit → PostgreSQL controller cleanup → owned
tunnel → vLLM cleanup.

Geniş process kill, Docker prune veya başka projelerin resource’larını silme
kullanılmaz. Ayrıntılar için:

- [PostgreSQL TLS controller](docs/runbooks/postgres_tls_service_controller.md)
- [vLLM service controller](docs/runbooks/vllm_e2e_service_controller.md)
- [Windows dashboard E2E](docs/runbooks/windows_dashboard_rag_vllm_e2e.md)

## Bilinen sınırlamalar

- Sentetik Demo RAG/vLLM’i tetiklemez; manuel uçtan uca UI yolu Yerel Ses
  Dosyasıdır.
- MiniLM doğrulaması teknik smoke’tur; Türkçe retrieval kalite kabulü değildir.
- Full E2E PostgreSQL lease’i `7200` saniyedir; bu formal total-runtime
  garantisi değildir.
- Gözlenen 15 KB WAV, ASR aşamasında başarısız olmuştur. Nedeni henüz
  doğrulanmamıştır ve açık takip maddesidir; RAG veya ingestion kusuru olarak
  sınıflandırılmamalıdır.
- Authentication, production deployment, OCR ve kalıcı orijinal belge saklama
  bu geliştirme kapsamının dışındadır.

Geliştirici devri için [docs/HANDOVER.md](docs/HANDOVER.md), kronoloji için
`docs/progress/`, operasyon için güncel runbook’lar kullanılmalıdır.
