# CallMetric Live ASR

CallMetric Live ASR, çağrı merkezi görüşmelerini düşük gecikmeyle yazıya çevirmeyi ve gerçek zamanlı temsilci koçluğunu desteklemeyi amaçlayan bir servistir.

## Mevcut durum

Proje şu anda başlangıç aşamasındadır. FastAPI uygulama iskeleti, sağlık kontrolü uç noktası ve bu uç noktayı doğrulayan temel test hazırdır. Canlı ses akışı ve ASR işlevleri henüz uygulanmamıştır.

## Gereksinimler

- Python 3.12 veya üzeri
- [uv](https://docs.astral.sh/uv/)

## Kurulum

Depoyu klonladıktan sonra proje klasöründe bağımlılıkları hazırlayın:

```shell
uv sync
```

## Sunucuyu çalıştırma

Geliştirme sunucusunu başlatın:

```shell
uv run uvicorn app.main:app --reload
```

Sunucu varsayılan olarak `http://127.0.0.1:8000` adresinde çalışır.

- Sağlık kontrolü: http://127.0.0.1:8000/health
- Swagger API belgeleri: http://127.0.0.1:8000/docs

## Testleri çalıştırma

```shell
uv run pytest
```

## Planlanan geliştirme aşamaları

1. Temel proje yapısı, belgeler ve kalite kontrolleri
2. Ses ön işleme bileşenleri
3. Canlı ses akışı yönetimi
4. ASR motoru entegrasyonu
5. Parçalı transkriptleri birleştirme
6. Gerçek zamanlı çağrı merkezi koçluğu için API geliştirme
7. Performans, gecikme ve doğruluk ölçümleri
8. Üretim ortamına hazırlık, izleme ve dağıtım
