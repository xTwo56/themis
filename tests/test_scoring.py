import pytest

from themis.scoring import score_classification_response


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ("Negative\n", "negative"),
        ("  POSITIVE ", "positive"),
        ("STRASSE", "straße"),
    ],
)
def test_score_classification_normalizes_whitespace_and_case(
    actual: str,
    expected: str,
) -> None:
    assert score_classification_response(actual, expected)


def test_score_classification_rejects_incorrect_response() -> None:
    assert not score_classification_response("positive", "negative")


def test_score_classification_rejects_verbose_response() -> None:
    assert not score_classification_response("The answer is negative", "negative")
