# Beyza Progress

## PR31 - Profile-Bound PostgreSQL Vector Upsert

Tarih: 28 Temmuz 2026

- `ProfileBoundPostgreSQLVectorStore.upsert(record: VectorRecord) -> None`
  eklendi.
- Record scope, canonical text, ordered metadata, exact profile dimension ve
  IEEE-754 float32 embedding transaction baslamadan dogrulanip complete stored
  row contract'ina donusturulur.
- Transaction sirasi scope lock, `for_update=True` profile read, complete
  expected-profile equality validation ve tek `replace_record` cagrisidir.
  Commit, rollback ve release transaction runner sorumlulugunda kalir.
- `replace_record` full tenant/knowledge-base/document/chunk identity icin
  transaction-local insert-or-replace semantigiyle belgelendi.
- Degisen dosyalar: `app/vector_store/postgres/adapter.py`,
  `app/vector_store/postgres/contracts.py`; eklenen dosyalar:
  `tests/test_postgres_upsert.py`, `docs/progress/beyza.md`.
- Focused PostgreSQL upsert/batch/boundary testleri: 163 passed.
- Full suite 1269 test toplarken mevcut Torch `c10.dll` WinError 1114 nedeniyle
  dort ASR/CLI/offline-evaluation testinin collection asamasinda durdu.
- Scoped Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: Profile-bound cosine search ayri PR32 kapsaminda
  ele alinacak; SQL, Psycopg ve runtime wiring ertelendi.

## PR32 - Profile-Bound PostgreSQL Cosine Search

Tarih: 28 Temmuz 2026

- `ProfileBoundPostgreSQLVectorStore.search(request: SearchRequest) ->
  SearchResult` eklendi ve adapter hem `VectorStore` hem
  `AtomicVectorBatchWriter` contract'larini structural olarak tamamlar.
- Query scope, top_k, minimum score ve embedding transaction baslamadan strict
  dogrulanir; query embedding IEEE-754 float32 canonical hale getirilir.
- Read-only transaction `for_update=False` profile read ve tek
  `search_cosine` cagrisidir; scope lock veya write kullanmaz.
- Returned row'lar threshold filtering'den once tamamen dogrulanir; duplicate
  identity ve top_k overflow fail-closed reddedilir. Sonuclar relevance
  descending, document ID ve chunk ID ile deterministic siralanir.
- Complete stored text, embedding ve ordered metadata provider-neutral
  `VectorRecord`/`VectorSearchHit` modellerine map edilir; native cosine
  distance result contract'ina sizmaz.
- Degisen dosyalar: `app/vector_store/postgres/adapter.py`,
  `app/vector_store/postgres/contracts.py`,
  `app/vector_store/postgres/__init__.py`, `tests/test_postgres_upsert.py`,
  `docs/progress/beyza.md`; eklenen dosya:
  `tests/test_postgres_cosine_search.py`.
- Focused PostgreSQL search/upsert/batch/boundary/retriever testleri:
  219 passed.
- Full suite 1323 test toplarken mevcut Torch `c10.dll` WinError 1114 nedeniyle
  dort ASR/CLI/offline-evaluation testinin collection asamasinda durdu.
- Sonraki planli adim: SQL/Psycopg transaction implementation'i, migrations ve
  runtime wiring ayri ve onayli PR33 kapsaminda ele alinacak.

## PR33 - Synchronous Psycopg PostgreSQL Transaction Runner

Tarih: 28 Temmuz 2026

- Injected connection ve transaction factory'leri kullanan stateless
  `PsycopgPostgreSQLVectorTransactionRunner` eklendi.
- Runner her operation icin tek fresh connection edinir; `autocommit=False`
  dogrulamasi, tek callback, success commit ve tum acquired-connection
  path'lerinde deterministic close lifecycle'i uygular.
- Primary transaction failure'lari ayni exception nesnesiyle korunur;
  rollback/close cleanup failure'lari rollback-then-close sirasiyla
  `ExceptionGroup` cause olarak saklanir.
- Dogrudan runtime dependency olarak `psycopg[binary]>=3.3.4,<4` eklendi.
- Degisen dosyalar: `pyproject.toml`, `uv.lock`, `docs/progress/beyza.md`;
  eklenen dosyalar: `app/vector_store/postgres/runner.py`,
  `tests/test_postgres_transaction_runner.py`.
- Focused PostgreSQL runner/search/upsert/batch/profile testleri: 168 passed.
- Full suite 1340 test toplarken mevcut Torch `c10.dll` WinError 1114 nedeniyle
  dort ASR/CLI/offline-evaluation testinin collection asamasinda durdu.
- Repository-wide Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: SQL transaction implementation'i, migration, Docker ve
  runtime wiring ayri onayli PR'lara ertelendi.

## PR34 - PostgreSQL/pgvector Vector-Store Migration

Tarih: 28 Temmuz 2026

- Forward-only `migrations/postgres/0001_vector_store.sql` eklendi.
- Migration tek transaction icinde vector extension, fixed
  `callmetric_vector` schema, migration ledger, embedding profile ve vector
  record tablolarini olusturur; ledger'a `0001` version'ini ekler.
- Profile scope/dimension uniqueness, cosine-only distance, full record
  identity, dimension-bearing restrictive foreign key, unbounded vector
  dimension/nonzero ve ordered metadata JSONB array constraints eklendi.
- Migration yalnizca statik olarak dogrulandi; SQL, PostgreSQL ve Docker
  calistirilmadi.
- Degisen dosya: `docs/progress/beyza.md`; eklenen dosyalar:
  `migrations/postgres/0001_vector_store.sql`,
  `tests/test_postgres_vector_migration.py`.
- Focused migration testleri: 21 passed.
- Full suite 1361 test toplarken mevcut Torch `c10.dll` WinError 1114 nedeniyle
  dort ASR/CLI/offline-evaluation testinin collection asamasinda durdu.
- Repository-wide Ruff check/format passed; Pyright 0 errors; conflict-marker
  script passed ve conflict-marker testleri 5 passed.

- Sonraki planli adim: complete Psycopg SQL transaction implementation'i ayri
  onayli PR35 kapsaminda ele alinacak.

## PR35 - Psycopg/pgvector Transaction Implementation

Tarih: 28 Temmuz 2026

- Lazy callable connection factory ile pgvector kaydi ve complete
  `PsycopgPostgreSQLVectorTransaction` implementasyonu eklendi.
- Scope lock, profile read/insert, record read/insert/replace ve cosine search
  olmak uzere mevcut yedi transaction operation'i parameterized SQL ile
  uygulandi; transaction lifecycle'i runner sorumlulugunda kaldi.
- Dogrudan runtime dependency olarak `pgvector>=0.5,<0.6` eklendi.
- Degisen dosyalar: `pyproject.toml`, `uv.lock`, `docs/progress/beyza.md`;
  eklenen dosyalar: `app/vector_store/postgres/connection_factory.py`,
  `app/vector_store/postgres/transaction.py`,
  `tests/test_pgvector_connection_factory.py`,
  `tests/test_postgres_transaction.py`.
- Focused PostgreSQL testleri: 242 passed.
- Full suite mevcut Torch `c10.dll` WinError 1114 nedeniyle dort
  ASR/CLI/offline-evaluation testinin collection asamasinda durdu.
- Ruff check/format passed; Pyright 0 errors; conflict-marker script passed ve
  conflict-marker testleri 5 passed.
- Sonraki planli adim: gercek PostgreSQL/Docker integration'i ayri ve onayli
  PR36 kapsaminda ele alinacak.

## PR36 - Docker-backed PostgreSQL/pgvector Integration

Tarih: 28 Temmuz 2026

- Docker 29.6.2 ve Docker Compose v5.3.1 ile
  `pgvector/pgvector:0.8.5-pg16-bookworm` image'i kullanildi.
- Checked-in `migrations/postgres/0001_vector_store.sql` fresh ve izole test
  database'ine basariyla uygulandi.
- Profile repository, profile-bound vector store, atomic batch, upsert, cosine
  search, rollback, scope isolation ve bounded advisory-lock davranisini
  kapsayan gercek PostgreSQL integration testleri: 9 passed.
- Unique Compose project'ine ait container, network ve volume runner tarafindan
  basariyla kaldirildi; cleanup sonrasinda PR36 kaynagi kalmadigi dogrulandi.
- Degisen dosyalar: `pyproject.toml`, `docs/progress/beyza.md`; eklenen
  dosyalar: `compose.postgres-integration.yml`,
  `scripts/run_postgres_integration.py`,
  `tests/integration/test_postgres_vector_integration.py`.
- Docker-disindaki full suite onceki kosuda 1423 test toplarken bilinen Torch
  `c10.dll` WinError 1114 nedeniyle dort ASR/CLI/offline-evaluation testinin
  collection asamasinda durdu; 9 integration testi opt-in olarak dislandi.
- Sonraki planli adim: production runtime configuration ve wiring ayri ve
  onayli PR37 kapsaminda ele alinacak.

## PR37 - Production PostgreSQL RAG Composition

Tarih: 28 Temmuz 2026

- SecretStr DSN, bounded connect timeout, secure SSL mode ve safe application
  name kullanan frozen `PostgreSQLVectorStoreSettings` eklendi.
- Tenant/knowledge-base, model identity/path, dimension, normalization,
  CPU/CUDA ve local-files-only politikasini acikca tasiyan frozen
  `KnowledgeBaseRAGProviderSettings` eklendi.
- Tek transaction runner ve tek lazy query/document embedder paylasan frozen
  `PostgreSQLRAGComposition` ile side-effect-free
  `compose_profile_bound_postgres_rag` factory'si eklendi.
- Composition sirasinda connection, pgvector registration, transaction,
  migration, profile registration, backend/model load, network veya GPU
  aktivitesi yapilmaz.
- Eklenen dosyalar: `app/composition/__init__.py`,
  `app/composition/postgres_rag.py`,
  `tests/test_postgres_rag_composition.py`; degisen dosya:
  `docs/progress/beyza.md`.
- Focused composition testleri: 43 passed; ilgili
  composition/PostgreSQL/embedding/ingestion/retrieval testleri: 500 passed.
- Full suite 1466 test toplarken mevcut Torch `c10.dll` WinError 1114 nedeniyle
  dort ASR/CLI/offline-evaluation testinin collection asamasinda durdu.
- Repository-wide Ruff check/format passed; Pyright 0 errors; conflict-marker
  script passed ve conflict-marker testleri 5 passed.

## PR41 - Reproducible Windows Torch Environment

Tarih: 29 Temmuz 2026

- Anaconda 3.12.4 tabanli eski `.venv` icinde Torch 2.13.0 native
  `c10.dll` import'u basarisizdi; proje Python pin'i uv-managed CPython
  3.12.12 olarak kesinlestirildi.
- Yalniz Windows icin direct `torch==2.12.0` runtime constraint'i eklendi.
  Lock'ta Windows CPython 3.12 `win_amd64` wheel'i 2.12.0, non-Windows Torch
  2.13.0 ve mevcut Linux CUDA dependency/marker'lari degismeden korundu.
- Eski ortam silinmeden
  `%TEMP%\callmetric-project-venv-backup-20260729-122706-082bc427e0324d5588e0b3cd9c1a8f41`
  konumuna tasindi; yeni locked ortam uv-managed CPython 3.12.12 kullaniyor.
- Torch 2.12.0+cpu import'u ve CPU tensor olusturma basarili; CUDA initialize
  edilmedi. CTranslate2, Faster-Whisper, SentenceTransformers ve SetFit
  modelsiz ayri subprocess'lerde basariyla import edildi.
- Pytest collection: 1645 collected; onceki dort failing modul: 55 passed;
  ilgili embedding/classification/ASR testleri: 184 passed.
- Full suite: 1634 passed, 11 opt-in PostgreSQL integration testi skipped,
  1 mevcut Starlette deprecation warning.
- Repository-wide Ruff check/format passed; Pyright 0 errors; `uv lock --check`
  ve conflict-marker script passed; conflict-marker testleri 5 passed.
- Sonraki planli adim: migration/schema readiness ve explicit profile
  provisioning ayri ve onayli PR38 kapsaminda ele alinacak.

## PR38 - PostgreSQL Schema Readiness

Tarih: 28 Temmuz 2026

- Side-effect-free constructor ve explicit `verify()` operation'i kullanan
  `PostgreSQLSchemaReadinessChecker` eklendi.
- Read-only transaction icinde exact pgvector `0.8.5`, fixed
  `callmetric_vector` schema, `0001` migration ledger kaydi, gerekli tablolar,
  kolonlar ve named constraint subset'leri fail-closed dogrulanir.
- Profile provisioning composition sirasinda otomatik yapilmaz; mevcut
  `composition.profile_repository.register_profile(composition.profile)`
  operation'i missing profile insert ve identical repeated registration icin
  explicit ve idempotent kalir.
- Gercek Docker PostgreSQL/pgvector integration testleri: 10 passed.
- Integration runner'a ait project-scoped container, network ve volume
  basariyla kaldirildi ve cleanup dogrulandi.
- Degisen dosyalar: `app/vector_store/postgres/__init__.py`,
  `tests/test_postgres_embedding_profile_repository.py`,
  `tests/test_postgres_vector_boundary.py`,
  `tests/integration/test_postgres_vector_integration.py`,
  `docs/progress/beyza.md`; eklenen dosyalar:
  `app/vector_store/postgres/readiness.py`,
  `tests/test_postgres_schema_readiness.py`.
- Focused readiness/profile/migration/PostgreSQL unit testleri: 401 passed.
- Full suite 1504 test toplarken mevcut Torch `c10.dll` WinError 1114 nedeniyle
  dort ASR/CLI/offline-evaluation testinin collection asamasinda durdu.
- Sonraki planli adim: deployment/runtime readiness invocation'i ayri ve
  ownership-onayli bir PR kapsaminda ele alinacak.

## PR39 - Explicit PostgreSQL RAG Profile Provisioning

Tarih: 29 Temmuz 2026

- Environment-backed secret PostgreSQL settings ve exact non-secret
  tenant/knowledge-base provider JSON contract'i kullanan explicit deployment
  operation'i ve thin CLI eklendi.
- Islem sirasi side-effect-free composition, read-only schema readiness ve
  explicit immutable profile registration'dir; returned canonical profile
  complete equality ile dogrulanir.
- CLI yalnizca `--provider-settings PATH` kabul eder; DSN ve connection
  ayarlari CLI/JSON'a alinmaz. Basari, configuration ve operational sonuclar
  fixed secret-safe mesajlarla sirasiyla 0, 2 ve 1 exit code'larina map edilir.
- Migration, automatic repair/startup wiring, model load/download, ingestion,
  retrieval, LLM, retry, logging ve pooling eklenmedi.
- Eklenen dosyalar: `app/deployment/__init__.py`,
  `app/deployment/postgres_rag.py`, `scripts/provision_postgres_rag.py`,
  `tests/test_postgres_rag_provisioning.py`,
  `tests/test_provision_postgres_rag_cli.py`; degisen dosya:
  `docs/progress/beyza.md`.
- Focused provisioning/CLI testleri: 34 passed; ilgili
  PostgreSQL/composition/readiness/profile testleri: 435 passed.
- Full suite 1538 test toplarken mevcut Torch `c10.dll` WinError 1114 nedeniyle
  dort ASR/CLI/offline-evaluation testinin collection asamasinda durdu.
- Repository-wide Ruff check/format passed; Pyright 0 errors.
- Sonraki planli adim: production deployment packaging/runbook ve operator
  invocation'i ayri, ownership-onayli bir PR kapsaminda ele alinacak.

## PR40 - Explicit PostgreSQL Vector Migration

Tarih: 29 Temmuz 2026

- Fixed `0001_vector_store.sql` registry, SHA-256 integrity validation,
  session advisory lock, fail-closed state inspection ve separate readiness
  verification kullanan explicit forward-only migration operation'i eklendi.
- Migration ve readiness connection'lari lock/statement timeout'larini yalniz
  strictly validated integer settings'ten uretilen exact libpq `options`
  kwarg'i ile alir; timeout SQL'i, configuration commit'i veya application
  import/startup migration'i eklenmedi.
- Migration file'in owned `BEGIN`/`COMMIT` siniri korunur; inspection rollback
  whole-file execution'dan once yapilir ve connection lifecycle explicit
  migration operation'i tarafindan yonetilir.
- Gercek Docker PostgreSQL 16.14 ve pgvector `0.8.5-pg16-bookworm`
  entegrasyonu: 11 passed; concurrent advisory-lock migration, idempotency,
  readiness ve mevcut vector-store akislarini dogruladi.
- Integration runner'a ait project-scoped container, network ve volume
  basariyla kaldirildi.
- Focused PostgreSQL/composition/embedding/ingestion/retrieval testleri:
  546 passed; son dogrudan PR40 testleri: 55 passed.
- Full suite bilinen Torch/CTranslate2 Windows DLL access-violation'i nedeniyle
  ASR/diarization test collection asamasinda durdu.
- Repository-wide Ruff check/format passed; Pyright 0 errors; conflict-marker
  script passed ve conflict-marker testleri 5 passed.

## PR43 - Explicit PostgreSQL RAG Chunk Ingestion

Tarih: 29 Temmuz 2026

- Preconstructed `DocumentIngestionRequest` chunk'larini alan explicit
  `ingest_profile_bound_postgres_rag` deployment operation'i ve thin CLI
  eklendi.
- Islem sirasi argument/scope validation, side-effect-free composition,
  read-only schema readiness, mutation yapmayan exact profile lookup ve
  sonrasinda lazy embedding ile atomic batch admission'dir.
- Migration veya profile registration/replacement otomatik yapilmaz; model
  readiness ve exact registered-profile dogrulamasindan once yuklenmez.
- CLI strict provider ve ingestion JSON contract'lari ile mevcut
  `CALLMETRIC_POSTGRES_*` environment ayarlarini kullanir; configuration,
  operational ve success ciktilari fixed ve secret-safe kalir.
- Eklenen dosyalar: `app/deployment/postgres_ingestion.py`,
  `scripts/ingest_postgres_rag.py`, `tests/test_postgres_rag_ingestion.py`,
  `tests/test_ingest_postgres_rag_cli.py`; degisen dosyalar:
  `app/deployment/__init__.py`, `tests/test_postgres_rag_provisioning.py`,
  `docs/progress/beyza.md`.
- Focused ingestion/deployment/composition/readiness/atomic testleri:
  175 passed.
- Full suite: 1678 passed, 11 opt-in PostgreSQL integration testi skipped,
  1 mevcut Starlette deprecation warning.
- Repository-wide Ruff check/format passed; Pyright 0 errors; conflict-marker
  script passed ve conflict-marker testleri 5 passed.
