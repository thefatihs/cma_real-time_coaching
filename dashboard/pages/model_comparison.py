import pandas as pd
import streamlit as st

from dashboard.common import (
    comparison_rows,
    filter_runs,
    load_runs,
    select_results_directory,
)


st.title("Model Karşılaştırma")
st.caption("Bu görünüm istatistiksel anlamlılık iddiasında bulunmaz.")

try:
    runs = load_runs(select_results_directory())
except Exception as error:
    st.error(f"Benchmark sonuçları yüklenemedi: {error}")
    st.stop()

experiments = sorted({run.experiment_id for run in runs})
models = sorted({run.model_name for run in runs})
experiment = st.selectbox("Deney", ["Tümü", *experiments])
model = st.selectbox("Model", ["Tümü", *models])
filtered = filter_runs(
    runs,
    experiment_id=None if experiment == "Tümü" else experiment,
    model_name=None if model == "Tümü" else model,
)
rows = comparison_rows(filtered)

if not rows:
    st.info("Seçilen filtreler için sonuç yok.")
    st.stop()

dataframe = pd.DataFrame(rows)
st.dataframe(dataframe, use_container_width=True, hide_index=True)
chart = dataframe.set_index("Model")[["Ağırlıklı WER"]]
st.bar_chart(chart)
