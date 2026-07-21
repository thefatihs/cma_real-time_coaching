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

## Offline ASR Baseline

İlk çevrimdışı ASR temeli Faster-Whisper kullanır. Yerel geliştirme ayarları, GPU'su olmayan geliştirme bilgisayarında çalışabilmesi için `tiny` model, `cpu` cihazı ve `int8` hesaplama türüdür.

Bu yapılandırma yalnızca geliştirme için bir başlangıç noktasıdır. `tiny`, projenin nihai doğruluk modeli değildir. Daha büyük ve daha doğru modeller ileride AWS üzerindeki GPU ortamında karşılaştırmalı olarak test edilecektir.

Mevcut modül tek bir ses dosyasını işler; henüz canlı akış desteği yoktur. Otomatik testler gerçek modeli sahte bir nesneyle değiştirir. Bu nedenle testler model dosyası indirmez, internet bağlantısı veya CUDA gerektirmez.
