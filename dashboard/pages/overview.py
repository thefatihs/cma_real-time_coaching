import streamlit as st

from app.benchmark.aggregation import aggregate_runs
from dashboard.common import comparison_rows, load_runs, select_results_directory


st.title("ASR Doğruluk Panosu")
st.caption("Daha düşük WER ve CER daha iyi transkripsiyon doğruluğu anlamına gelir.")

try:
    runs = load_runs(select_results_directory())
except Exception as error:
    st.error(f"Benchmark sonuçları yüklenemedi: {error}")
    st.stop()

if not runs:
    st.info("Henüz kayıtlı benchmark sonucu yok.")
    st.stop()

metrics = aggregate_runs(runs)
best_rows = comparison_rows(runs)
columns = st.columns(4)
columns[0].metric("Toplam çalıştırma", metrics.run_count)
columns[1].metric("Referans kelime", metrics.total_reference_words)
columns[2].metric(
    "Ağırlıklı WER",
    f"{metrics.weighted_wer:.2%}" if metrics.weighted_wer is not None else "—",
)
columns[3].metric(
    "Ağırlıklı CER",
    f"{metrics.weighted_cer:.2%}" if metrics.weighted_cer is not None else "—",
)

best = best_rows[0]
st.subheader("En iyi model/yapılandırma")
st.write(
    f"{best['Model']} · beam={best['Beam']} · VAD={best['VAD']} · "
    f"önceki metin={best['Önceki metin']}"
)

error_columns = st.columns(3)
error_columns[0].metric("Değiştirme", metrics.total_substitutions)
error_columns[1].metric("Silme", metrics.total_deletions)
error_columns[2].metric("Ekleme", metrics.total_insertions)
