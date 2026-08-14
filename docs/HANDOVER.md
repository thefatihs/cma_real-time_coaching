# CallMetric Geliştirici Devri

Bu belge `feat/dashboard-rag-document-upload` hattındaki mimariyi, güvenlik
sınırlarını ve operasyon devrini özetler. Exact komut ve environment
sözleşmelerinin otoritatif kaynağı runbook’lardır; gizli veya makineye özgü
değerler burada tekrarlanmaz.

## Teslim edilen mimari

- Tenant-aware ASR, SetFit sınıflandırma ve deterministik koçluk.
- Streamlit temsilci görünümü ve **Bilgi Tabanı** belge yönetimi.
- Bounded PDF/TXT/Markdown hazırlama ve arka plan ingestion.
- Migration `0001`–`0003`, PostgreSQL registry/job ve pgvector retrieval.
- Offline CPU MiniLM ile 384 boyutlu normalize embedding.
- HTTPS vLLM, structured validation, result gate ve güvenli citation.
- Exact-owned PostgreSQL/vLLM controller lifecycle ve secret-safe fazlar.

## Belge ingestion yaşam döngüsü

1. Tenant ve knowledge-base seçili `TenantConfig` ile doğrulanmış provider
   ayarlarından gelir; upload alanı bunları belirleyemez.
2. Upload boundary dosya türü, boyut, ad, encoding ve PDF güvenliğini doğrular;
   sunucu kimlikleri ile submission token üretir.
3. Kaynak byte’ları bounded bellek içi envelope’da tutulur; kalıcı dosya yolu
   veya kaynak nesnesi oluşturulmaz.
4. Registry belge ile `QUEUED` job’ı atomik oluşturur. Aynı scoped SHA-256
   mevcut belgeye idempotent olarak çözülür.
5. Worker extraction, deterministic chunking ve MiniLM embedding yapar.
6. Vector admission ile registry finalization aynı PostgreSQL transaction’ında
   tamamlanır; başarı `READY`, güvenli sabit fazlı hata `FAILED` olur.
7. Başarı, hata, iptal ve close sonrasında kaynak byte’ları bırakılır.

Schema ve migration ayrıntıları için
[PostgreSQL RAG runbook](runbooks/postgres_rag_smoke.md) esas alınır.

## Retrieval, coaching ve citation

1. Yerel ses ASR ile transcript event’lerine dönüştürülür.
2. Checksum-uyumlu SetFit artifact’i canonical classification label üretir.
3. Decision gate yalnız policy’nin izin verdiği olayı bounded RAG manager’a
   gönderir.
4. MiniLM query embedding’i profile uygun üretilir; pgvector araması exact
   tenant + knowledge-base kapsamında yapılır.
5. Boş retrieval vLLM’i çağırmaz; dolu retrieval yalnız izinli citation
   kimliklerini deterministic prompt’a taşır.
6. HTTPS vLLM çıktısı schema, scope, evidence ve grounding gate’lerinden
   geçmeden dashboard’a alınmaz.
7. Server-side projector iç kimlikleri güvenli, immutable sunum nesnelerine
   çevirir. UI yalnız dosya adı, tür ve varsa güvenilir sayfa bilgisini gösterir.

## Dashboard davranışı

- **Sentetik Demo** production PostgreSQL retrieval veya vLLM zincirini
  tetiklemez.
- Desteklenen manuel RAG yolu **Yerel Ses Dosyası**dır; ASR/SetFit ve hazır
  PostgreSQL, MiniLM, HTTPS vLLM gerektirir.
- **Bilgi Tabanı** upload, progress, tenant-scoped listeleme, iptal ve onaylı
  exact deletion sunar.
- `READY` belge retrieval kanıtı değildir; citation yalnız accepted grounded
  sonuçta gösterilir.
- RAG unavailable olduğunda temel ASR/sınıflandırma/kural koçluğu çalışır;
  upload hatası aktif çağrıyı durdurmaz.

## Model sözleşmeleri

### MiniLM

- Production canonical model kimliğini destekler. Manuel/E2E kullanımında ayrıca
  onaylı immutable local snapshot kullanılabilir.
- Local snapshot yalnız ignored artifact root içinde tutulur; manifest üyeliği,
  dosya boyutları ve SHA-256 değerleri model kurulmadan önce doğrulanır.
- Canonical profile 384 boyutlu, finite, normalize ve cosine uyumludur. Local
  snapshot CPU, local-only ve remote-code kapalı biçimde yüklenir.
- Makineye özgü snapshot yolu PostgreSQL, UI, session state veya loga yazılmaz.
- Teknik doğrulama Türkçe retrieval kalite kabulü değildir.

### SetFit ve ASR

- SetFit metadata, taxonomy ve threshold profile checksum’ları eşleşmelidir.
- Artifact yok/uyumsuz olduğunda sabit güvenli fallback uygulanır.
- Yerel Ses Dosyası production RAG yolu ASR gerektirir; Sentetik Demo bunun
  yerine geçmez.
- Gözlenen 15 KB WAV ASR aşamasında başarısız olmuştur. Kök neden henüz
  doğrulanmamıştır ve açık takip maddesidir.

### HTTPS vLLM

- Yalnız HTTPS, doğrulanmış CA/hostname ve bounded timeout/output kabul edilir.
- Endpoint/model ayarı/token yalnız environment veya private handoff’tan gelir.
- Raw provider/database exception’ları UI veya loga yansıtılmaz.

## Environment değişkenleri

| Alan | İsimler |
| --- | --- |
| Tenant | `CALLMETRIC_DASHBOARD_SMOKE_TENANT_OVERRIDE_PATH` |
| RAG | `CALLMETRIC_DASHBOARD_RAG_PROVIDER_SETTINGS_PATH`, `CALLMETRIC_DASHBOARD_RAG_INTEGRATION_POLICY_PATH`, `CALLMETRIC_DASHBOARD_RAG_MAX_WORKERS`, `CALLMETRIC_DASHBOARD_RAG_CAPACITY` |
| Belge | `CALLMETRIC_DASHBOARD_DOCUMENT_MAX_WORKERS`, `CALLMETRIC_DASHBOARD_DOCUMENT_CAPACITY` |
| PostgreSQL | `CALLMETRIC_POSTGRES_DSN`, `CALLMETRIC_POSTGRES_CONNECT_TIMEOUT_SECONDS`, `CALLMETRIC_POSTGRES_SSL_MODE`, `CALLMETRIC_POSTGRES_APPLICATION_NAME` |
| Migration | `CALLMETRIC_POSTGRES_MIGRATION_*` |
| TLS service | `CALLMETRIC_POSTGRES_TLS_SERVICE_*` |
| vLLM | `CALLMETRIC_VLLM_BASE_URL`, `CALLMETRIC_VLLM_MODEL_ID`, `CALLMETRIC_VLLM_API_TOKEN`, `CALLMETRIC_VLLM_CA_CERTIFICATE_PATH`, `CALLMETRIC_VLLM_CONNECT_TIMEOUT_SECONDS`, `CALLMETRIC_VLLM_READ_TIMEOUT_SECONDS`, `CALLMETRIC_VLLM_MAX_OUTPUT_TOKENS`, `CALLMETRIC_VLLM_TEMPERATURE`, `CALLMETRIC_VLLM_VERIFY_TLS` |

Değer, DSN, token, sertifika içeriği, private absolute path, müşteri verisi veya
environment dökümü dokümana ve Git’e eklenmez.

## Doğrulanmış gerçek E2E

Gerçek Windows document-dashboard RAG/vLLM çalışması tam olarak `E2E_OK`
üretmiştir. TLS PostgreSQL/pgvector, migration/readiness, document `READY`,
gerçek pinned MiniLM embedding, scoped retrieval, validated tunnel üzerinden
Ubuntu HTTPS vLLM, result gate, güvenli citation, duplicate replay, exact
deletion/scope isolation ve kalıntısız cleanup doğrulanmıştır.

Full E2E PostgreSQL service lease’i exact `7200` ister. Bu desteklenen operator
penceresinin maksimum lease değeridir. Standalone PostgreSQL ve
`--postgres-startup-only` kontrolü `300`–`7200` aralığını kabul eder. `7200`,
bütün model, embedding ve SQL işlemlerini kapsayan formal toplam çalışma süresi
garantisi değildir.

## Güvenli başlatma ve kapatma

Başlatma:

1. [vLLM controller](runbooks/vllm_e2e_service_controller.md) ile HTTPS vLLM’i
   READY durumuna getir.
2. Doğrulanmış ve yalnız loopback’e bağlanan owned SSH tunnel’ı kur. Tunnel
   vLLM’i internete açmamalıdır.
3. [PostgreSQL TLS controller](runbooks/postgres_tls_service_controller.md) ile
   disposable PostgreSQL/pgvector.
4. Private handoff’u yalnız dashboard terminaline aktar.
5. Tenant/provider/policy ve worker ayarlarını doğrulayıp Streamlit’i başlat.
6. READY/generation için [RAG runbook](runbooks/postgres_rag_smoke.md)’u izle.

Kapatma:

1. Dashboard Reset ile çağrı ve belge resource’larını kapat.
2. Streamlit’i kontrollü durdur ve çıkmasını bekle.
3. PostgreSQL controller’ı durdur; exact project/handoff cleanup’ını bekle.
4. Yalnız owned tunnel’ı kapat.
5. vLLM controller’ı durdur; exact resource cleanup’ını doğrula.

Process adına göre kill, sabit PID, Docker prune, prefix tabanlı silme veya
başka projelerin resource’larına müdahale kullanılmaz.

## Güvenlik kontrol listesi

- Tenant/knowledge-base yalnız trusted server configuration’dan gelir.
- Original filename yalnız display metadata’dır, path değildir.
- Belge byte’ları, transcript, prompt, completion ve embedding loglanmaz.
- UI’da iç kimlik, SHA-256, storage key veya raw chunk gösterilmez.
- TLS verification kapatılmaz; public vLLM exposure kullanılmaz.
- Model artifact’leri ignored local artifact alanında kalır.
- Testlerde yalnız sentetik veri kullanılır.
- Cleanup yalnız exact owned process/resource/handoff’u etkiler.

## Bilinen sınırlamalar ve takip

- Sentetik Demo RAG/vLLM’i tetiklemez.
- Manuel UI smoke Yerel Ses Dosyası ve çalışan ASR/SetFit artifact’lerine
  bağlıdır.
- 15 KB WAV ASR başarısızlığının nedeni açık takip maddesidir.
- OCR, authentication, production deployment ve kalıcı kaynak saklama yoktur.
- `7200` lease formal uçtan uca deadline değildir.

## Doğrulama

```shell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv lock --check
uv run python scripts/check_conflict_markers.py
git diff --check
```

Gerçek PostgreSQL/vLLM smoke opt-in’dir. Sonraki açık teknik iş, 15 KB WAV ASR
başarısızlığını sentetik ve secret-safe bir yeniden üretimle izole etmektir.
