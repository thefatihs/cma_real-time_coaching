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
