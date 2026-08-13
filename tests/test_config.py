from pathlib import Path

import pytest
from pydantic import ValidationError

from themis.config import (
    ExecutionConfig,
    load_benchmark,
    load_dataset,
)


def test_execution_config_rejects_zero_repetitions() -> None:
    with pytest.raises(ValidationError):
        ExecutionConfig(
            warmup_runs=0,
            repetitions=0,
        )


def test_load_benchmark_returns_valid_spec(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.yaml"

    benchmark_path.write_text(
        """
name: test-benchmark

models:
  - model-a
  - model-b

prompt_template: |
  Classify this text: {input}

generation:
  temperature: 0.0
  max_output_tokens: 8
  context_length: 4096

execution:
  warmup_runs: 1
  repetitions: 3

dataset: datasets/test.jsonl
scorer: classification
""".strip(),
        encoding="utf-8",
    )

    benchmark = load_benchmark(benchmark_path)

    assert benchmark.name == "test-benchmark"
    assert benchmark.models == ["model-a", "model-b"]
    assert benchmark.execution.warmup_runs == 1
    assert benchmark.execution.repetitions == 3


def test_load_dataset_returns_valid_cases(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"

    dataset_path.write_text(
        "\n".join(
            [
                '{"id":"case-001","input":"It arrived.","expected":"neutral"}',
                '{"id":"case-002","input":"I love it.","expected":"positive"}',
            ]
        ),
        encoding="utf-8",
    )

    cases = load_dataset(dataset_path)

    assert len(cases) == 2

    assert cases[0].id == "case-001"
    assert cases[0].input == "It arrived."
    assert cases[0].expected == "neutral"

    assert cases[1].id == "case-002"
    assert cases[1].input == "I love it."
    assert cases[1].expected == "positive"


@pytest.mark.parametrize(
    "document",
    [
        '{"input":"I love it.","expected":"positive"}',
        '{"id":"case-001","expected":"positive"}',
        '{"id":"case-001","input":"I love it."}',
        '{"id":"case-001","input":"","expected":"positive"}',
        '{"id":"case-001","input":"I love it.","expected":""}',
    ],
)
def test_load_dataset_rejects_malformed_cases(
    tmp_path: Path,
    document: str,
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(document, encoding="utf-8")

    with pytest.raises(ValidationError):
        load_dataset(dataset_path)


def test_load_dataset_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                '{"id":"case-001","input":"First","expected":"yes"}',
                '{"id":"case-001","input":"Second","expected":"no"}',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_dataset(dataset_path)
