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

## Manual Offline Transcription

Elle denemek istediğiniz yerel ses dosyalarını `samples/` klasörüne koyun. Bu klasördeki ses dosyaları Git tarafından yok sayılır; yalnızca klasörü depoda tutan `.gitkeep` dosyası takip edilir.

İlk gerçek çalıştırma, seçilen Faster-Whisper modelini indirebilir. Varsayılan geliştirme ayarları `tiny` model, `cpu` cihazı ve `int8` hesaplama türüdür.

M4A örneği:

```shell
uv run python scripts/transcribe_file.py samples/deneme.m4a
```

WAV örneği ve isteğe bağlı ayarlar:

```shell
uv run python scripts/transcribe_file.py samples/deneme.wav --model tiny --language tr --beam-size 1 --cpu-threads 4
```

## Accuracy Evaluation

Model doğruluğunu anlamlı biçimde ölçmek için insan tarafından dinlenip doğrulanmış referans transkriptler gerekir. ASR çıktısı bu güvenilir metinle karşılaştırılır:

- WER (Word Error Rate), kelime düzeyindeki değiştirme, silme ve ekleme hatalarını ölçer.
- CER (Character Error Rate), aynı karşılaştırmayı karakter düzeyinde yapar.

Ham çağrı merkezi kayıtları ve özel referans transkriptleri Git deposunun dışında kalmalıdır. Değerlendirme aracı bu harici UTF-8 metin dosyalarının yollarını yalnızca çalışma anında kabul eder:

```shell
uv run python scripts/evaluate_transcript.py --reference C:\CallMetricPrivate\references\call_001.txt --hypothesis C:\CallMetricPrivate\hypotheses\call_001.txt
```

Yerel `tiny`, CPU ve `int8` yapılandırması yalnızca işlevsel bir başlangıçtır; nihai doğruluk modeli değildir. Son model seçimi, gerçek çağrı merkezi verileri kullanılarak AWS GPU ortamında yapılacak karşılaştırmalı doğruluk ölçümlerine dayanacaktır. Bu aşamada öncelik doğruluktur; gecikme optimizasyonu daha sonra yapılacaktır.

## Audio Metadata Inspection

Özel çağrı merkezi kayıtları Git deposunun dışında kalmalıdır. Metadata aracı yalnızca konteyner ve ilk ses akışındaki teknik bilgileri okur; ses framelerini tamamen decode etmez, oynatmaz veya konuşmayı yazıya çevirmez. Dosya içeriğini ve tam özel dosya yolunu ekrana yazdırmaz.

Windows PowerShell örneği:

```powershell
uv run python scripts/inspect_audio.py C:\CallMetricPrivate\audio\call_001.wav
```

Çıktı; dosya uzantısı, konteyner, codec, süre, örnekleme hızı, kanal bilgileri, örnek formatı ve varsa bit hızını gösterir.

## Accuracy-focused Offline Experiments

Uzun kayıtlardaki tekrar eden halüsinasyonları kontrollü biçimde incelemek için önce kısa bir WAV parçası çıkarabilirsiniz. Araç kaynak örnekleme hızını ve mono kanal düzenini korur; çıktıyı PCM signed 16-bit WAV olarak yazar:

```powershell
uv run python scripts/extract_audio_segment.py `
  "C:\CallMetricPrivate\audio\call_001.wav" `
  --start 30 `
  --end 75 `
  --output "C:\CallMetricPrivate\audio\call_001_0030_0115.wav"
```

Doğruluk deneylerinde VAD, önceki metne koşullama ve başlangıç prompt'u ayrı ayrı değiştirilebilir. Temiz transkripti harici bir UTF-8 dosyasına yazmak için:

```powershell
uv run python scripts/transcribe_file.py `
  "C:\CallMetricPrivate\audio\call_001_0030_0115.wav" `
  --model base `
  --vad-filter `
  --no-condition-on-previous-text `
  --initial-prompt "Çağrı merkezi müşteri görüşmesi." `
  --output-text "C:\CallMetricPrivate\hypotheses\call_001_base.txt"
```

`--output-text` dosyası yalnız temiz tam transkripti içerir; ölçümler ve terminal başlıkları eklenmez. Özel ses ve hipotez dosyaları Git deposunun dışında tutulmalıdır.

## Benchmark Result Recording

İnsan onaylı referans ile mevcut ASR hipotezini karşılaştırıp gizlilik güvenli bir benchmark sonucu kaydetmek için:

```powershell
uv run python scripts/record_benchmark_result.py `
  --reference "C:\CallMetricPrivate\references\call_001_0030_0115_reference.txt" `
  --hypothesis "C:\CallMetricPrivate\hypotheses\call_001_large_v3_vad_noctx.txt" `
  --results-dir "C:\CallMetricPrivate\results" `
  --experiment-id "call001-large-v3-baseline" `
  --recording-id "call_001" `
  --segment-id "0030-0115" `
  --start 30 --end 75 --model large-v3 --language tr --beam-size 5 `
  --vad-filter --no-condition-on-previous-text `
  --device cpu --compute-type int8 --cpu-threads 4
```

Her çalıştırma bir JSON dosyasına yazılır; `benchmark_runs.csv` bu JSON dosyalarından yeniden oluşturulur. Kalıcı sonuçlar yalnız metrikleri, mantıksal kimlikleri ve dosya adlarını içerir. Transkript metni ve mutlak özel dosya yolları kaydedilmez.

Toplu WER, çalıştırma yüzdelerinin basit ortalaması değildir: toplam kelime hatası toplam referans kelime sayısına bölünür. CER de toplam karakter hatasının toplam referans karakter sayısına bölünmesiyle aynı şekilde ağırlıklı hesaplanır.

## Accuracy Dashboard

Harici sonuç klasörünü PowerShell oturumunda seçmek için:

```powershell
$env:CALLMETRIC_BENCHMARK_DIR = "C:\CallMetricPrivate\results"
```

Dashboard'u başlatın:

```shell
uv run streamlit run dashboard/app.py
```

Streamlit normalde yerel adres olarak `http://localhost:8501` gösterir. Dashboard ASR çalıştırmaz; mevcut JSON ve CSV metriklerini yerel olarak, salt-okunur biçimde gösterir. Özel kayıtlar ve transkriptler Git dışında kalmalıdır.
