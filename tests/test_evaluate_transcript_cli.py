from pathlib import Path

import pytest

from scripts.evaluate_transcript import main


def test_successful_cli_output(capsys: pytest.CaptureFixture[str]) -> None:
    texts = {
        "Reference": "MÜŞTERİ, bugün aradı.",
        "Hypothesis": "müşteri dün aradı",
    }

    def text_reader(path: Path, label: str) -> str:
        return texts[label]

    exit_code = main(
        ["--reference", "private/reference.txt", "--hypothesis", "private/asr.txt"],
        text_reader=text_reader,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Normalized reference: müşteri bugün aradı" in captured.out
    assert "Normalized hypothesis: müşteri dün aradı" in captured.out
    assert "WER: 33.33%" in captured.out
    assert "Substitutions: 1" in captured.out
    assert "Deletions: 0" in captured.out
    assert "Insertions: 0" in captured.out
    assert "Correct words: 2" in captured.out
    assert captured.err == ""


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("Reference file not found"),
        ValueError("Reference path is a directory"),
        ValueError("Reference transcript is empty after normalization"),
        UnicodeDecodeError("utf-8", b"x", 0, 1, "invalid encoding"),
    ],
)
def test_cli_failures_return_nonzero_exit_code(
    error: Exception, capsys: pytest.CaptureFixture[str]
) -> None:
    def failing_reader(path: Path, label: str) -> str:
        raise error

    exit_code = main(
        ["--reference", "reference.txt", "--hypothesis", "hypothesis.txt"],
        text_reader=failing_reader,
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Error:" in captured.err
    assert captured.out == ""
