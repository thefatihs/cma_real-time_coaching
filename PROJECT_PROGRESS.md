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