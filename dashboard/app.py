import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


st.set_page_config(page_title="CallMetric ASR Doğruluk", layout="wide")

navigation = st.navigation(
    [
        st.Page("pages/overview.py", title="Genel Bakış", icon="📊", default=True),
        st.Page("pages/model_comparison.py", title="Model Karşılaştırma", icon="⚖️"),
        st.Page("pages/run_history.py", title="Çalıştırma Geçmişi", icon="🧾"),
        st.Page("pages/single_evaluation.py", title="Tek Değerlendirme", icon="📝"),
    ]
)
navigation.run()
