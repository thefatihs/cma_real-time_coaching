import streamlit as st

from app.evaluation.metrics import evaluate_transcript


st.title("Tek Transkript Değerlendirmesi")
st.info("Bu sayfaya yapıştırılan metinler bellekte değerlendirilir ve kaydedilmez.")

reference = st.text_area("İnsan onaylı referans")
hypothesis = st.text_area("ASR hipotezi")

if st.button("Değerlendir", type="primary"):
    try:
        result = evaluate_transcript(reference, hypothesis)
    except ValueError as error:
        st.error(str(error))
    else:
        st.write("Normalleştirilmiş referans:", result.normalized_reference)
        st.write("Normalleştirilmiş hipotez:", result.normalized_hypothesis)
        columns = st.columns(2)
        columns[0].metric("WER", f"{result.wer:.2%}")
        columns[1].metric("CER", f"{result.cer:.2%}")
        st.write(
            {
                "Değiştirme": result.substitutions,
                "Silme": result.deletions,
                "Ekleme": result.insertions,
                "Doğru kelime": result.correct_words,
            }
        )
