def score_classification_response(actual: str, expected: str) -> bool:
    return actual.strip().casefold() == expected.strip().casefold()
