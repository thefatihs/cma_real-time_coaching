from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_empty_state_smoke(tmp_path: Path, monkeypatch: object) -> None:
    app_path = Path(__file__).parents[1] / "dashboard" / "app.py"
    test = AppTest.from_file(str(app_path), default_timeout=10)

    test.run()

    assert not test.exception
