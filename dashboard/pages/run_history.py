import pandas as pd
import streamlit as st

from app.benchmark.repository import BenchmarkRepository
from dashboard.common import filter_runs, load_runs, select_results_directory


st.title("Çalıştırma Geçmişi")
results_directory = select_results_directory()

try:
    runs = load_runs(results_directory)
except Exception as error:
    st.error(f"Benchmark sonuçları yüklenemedi: {error}")
    st.stop()

experiments = sorted({run.experiment_id for run in runs})
models = sorted({run.model_name for run in runs})
experiment = st.selectbox("Deney", ["Tümü", *experiments])
model = st.selectbox("Model", ["Tümü", *models])
sort_order = st.selectbox("Sıralama", ["En yeni", "En düşük WER"])
filtered = filter_runs(
    runs,
    experiment_id=None if experiment == "Tümü" else experiment,
    model_name=None if model == "Tümü" else model,
)
filtered.sort(
    key=(lambda run: run.created_at_utc)
    if sort_order == "En yeni"
    else (lambda run: run.wer),
    reverse=sort_order == "En yeni",
)

rows = [
    {
        "Kayıt": run.recording_id,
        "Segment": run.segment_id,
        "Model": run.model_name,
        "WER": run.wer,
        "CER": run.cer,
        "Değiştirme": run.substitutions,
        "Silme": run.deletions,
        "Ekleme": run.insertions,
        "Süre (sn)": run.duration_seconds,
        "Oluşturulma": run.created_at_utc,
    }
    for run in filtered
]
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

summary_path = results_directory / BenchmarkRepository.SUMMARY_FILENAME
if summary_path.is_file():
    st.download_button(
        "Mevcut CSV özetini indir",
        data=summary_path.read_bytes(),
        file_name=BenchmarkRepository.SUMMARY_FILENAME,
        mime="text/csv",
    )
else:
    st.caption("İndirilebilir CSV özeti henüz oluşturulmamış.")
