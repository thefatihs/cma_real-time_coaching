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
