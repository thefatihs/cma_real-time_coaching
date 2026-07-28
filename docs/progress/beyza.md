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
