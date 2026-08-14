from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from themis.config import BenchmarkCase, BenchmarkSpec
from themis.runner import (
    BenchmarkResult,
    ModelMeasurement,
    aggregate_performance,
    aggregate_quality,
    execute_benchmark,
    export_benchmark_result,
    run_model,
)


class StubResponse:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self.result


class StubClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.requests: list[dict[str, Any]] = []

    def post(self, url: str, json: dict[str, Any]) -> StubResponse:
        self.requests.append({"url": url, "json": json})
        return StubResponse(make_raw_response(next(self.responses)))


def make_raw_response(response_text: str) -> dict[str, Any]:
    return {
        "response": response_text,
        "total_duration": 20,
        "load_duration": 1,
        "prompt_eval_count": 3,
        "prompt_eval_duration": 10,
        "eval_count": 2,
        "eval_duration": 5,
    }


def make_benchmark(*, repetitions: int = 1) -> BenchmarkSpec:
    return BenchmarkSpec.model_validate(
        {
            "name": "test",
            "models": ["model-a", "model-b"],
            "prompt_template": "Classify: {input}",
            "generation": {
                "temperature": 0.0,
                "max_output_tokens": 8,
                "context_length": 4096,
            },
            "execution": {"warmup_runs": 0, "repetitions": repetitions},
            "dataset": "dataset.jsonl",
            "scorer": "classification",
        }
    )


def make_measurement(
    model_name: str,
    is_correct: bool,
    *,
    client_duration_ns: int = 30,
    ollama_duration_ns: int = 20,
    load_duration_ns: int = 1,
    prompt_token_count: int = 3,
    prompt_duration_ns: int = 10,
    output_token_count: int = 2,
    output_duration_ns: int = 5,
) -> ModelMeasurement:
    raw_response = make_raw_response("positive")
    return ModelMeasurement(
        model_name=model_name,
        case_id="case-001",
        repetition=1,
        response_text="positive",
        is_correct=is_correct,
        client_duration_ns=client_duration_ns,
        ollama_duration_ns=ollama_duration_ns,
        load_duration_ns=load_duration_ns,
        prompt_token_count=prompt_token_count,
        prompt_duration_ns=prompt_duration_ns,
        output_token_count=output_token_count,
        output_duration_ns=output_duration_ns,
        raw_response=raw_response,
    )


def test_run_model_propagates_case_id_and_correctness() -> None:
    client = StubClient([" Positive\n"])
    case = BenchmarkCase(id="case-123", input="Great", expected="positive")

    measurement = run_model(client, make_benchmark(), case, "model-a", 2)

    assert measurement.case_id == "case-123"
    assert measurement.is_correct is True
    assert measurement.response_text == " Positive\n"
    assert measurement.repetition == 2
    assert measurement.raw_response["response"] == " Positive\n"


def test_execute_benchmark_runs_each_case_model_and_repetition() -> None:
    client = StubClient(["yes"] * 8)
    cases = [
        BenchmarkCase(id="one", input="First", expected="yes"),
        BenchmarkCase(id="two", input="Second", expected="no"),
    ]

    measurements = execute_benchmark(
        client,
        make_benchmark(repetitions=2),
        cases,
    )

    assert len(measurements) == 8
    assert sum(measurement.is_correct for measurement in measurements) == 4
    assert {measurement.model_name for measurement in measurements} == {
        "model-a",
        "model-b",
    }


def test_aggregate_quality_counts_all_measurements() -> None:
    summaries = aggregate_quality(
        [
            make_measurement("model-a", True),
            make_measurement("model-a", False),
            make_measurement("model-a", True),
        ]
    )

    assert len(summaries) == 1
    assert summaries[0].evaluated_run_count == 3
    assert summaries[0].correct_run_count == 2
    assert summaries[0].accuracy == 2 / 3


def test_aggregate_quality_separates_models() -> None:
    summaries = aggregate_quality(
        [
            make_measurement("model-a", True),
            make_measurement("model-b", False),
            make_measurement("model-b", False),
        ]
    )

    by_model = {summary.model_name: summary for summary in summaries}
    assert by_model["model-a"].accuracy == 1.0
    assert by_model["model-b"].accuracy == 0.0
    assert by_model["model-b"].evaluated_run_count == 2


def test_aggregate_quality_handles_empty_input() -> None:
    assert aggregate_quality([]) == []


def test_aggregate_performance_calculates_mean_and_median() -> None:
    measurements = [
        make_measurement(
            "model-a",
            True,
            client_duration_ns=10,
            ollama_duration_ns=100,
            load_duration_ns=1,
            prompt_token_count=1,
            prompt_duration_ns=1_000_000_000,
            output_token_count=2,
            output_duration_ns=1_000_000_000,
        ),
        make_measurement(
            "model-a",
            False,
            client_duration_ns=20,
            ollama_duration_ns=200,
            load_duration_ns=2,
            prompt_token_count=3,
            prompt_duration_ns=1_000_000_000,
            output_token_count=4,
            output_duration_ns=1_000_000_000,
        ),
        make_measurement(
            "model-a",
            True,
            client_duration_ns=60,
            ollama_duration_ns=600,
            load_duration_ns=6,
            prompt_token_count=8,
            prompt_duration_ns=1_000_000_000,
            output_token_count=9,
            output_duration_ns=1_000_000_000,
        ),
    ]

    summary = aggregate_performance(measurements)[0]

    assert summary.measured_run_count == 3
    assert summary.client_duration_ns.mean == 30
    assert summary.client_duration_ns.median == 20
    assert summary.ollama_duration_ns.mean == 300
    assert summary.ollama_duration_ns.median == 200
    assert summary.load_duration_ns.mean == 3
    assert summary.load_duration_ns.median == 2
    assert summary.prompt_tokens_per_second is not None
    assert summary.prompt_tokens_per_second.mean == 4
    assert summary.prompt_tokens_per_second.median == 3
    assert summary.output_tokens_per_second is not None
    assert summary.output_tokens_per_second.mean == 5
    assert summary.output_tokens_per_second.median == 4


def test_aggregate_performance_groups_models() -> None:
    summaries = aggregate_performance(
        [
            make_measurement("model-a", True),
            make_measurement("model-b", False),
            make_measurement("model-b", True),
        ]
    )

    by_model = {summary.model_name: summary for summary in summaries}
    assert by_model["model-a"].measured_run_count == 1
    assert by_model["model-b"].measured_run_count == 2


def test_aggregate_performance_ignores_none_throughput() -> None:
    summaries = aggregate_performance(
        [
            make_measurement(
                "model-a",
                True,
                prompt_token_count=10,
                prompt_duration_ns=0,
                output_duration_ns=0,
            ),
            make_measurement(
                "model-a",
                True,
                prompt_token_count=5,
                prompt_duration_ns=1_000_000_000,
                output_duration_ns=0,
            ),
        ]
    )

    assert summaries[0].prompt_tokens_per_second is not None
    assert summaries[0].prompt_tokens_per_second.mean == 5
    assert summaries[0].prompt_tokens_per_second.median == 5
    assert summaries[0].output_tokens_per_second is None


def test_aggregate_performance_handles_empty_input() -> None:
    assert aggregate_performance([]) == []


def test_benchmark_result_preserves_all_result_sections() -> None:
    measurements = [make_measurement("model-a", True)]
    quality_summaries = aggregate_quality(measurements)
    performance_summaries = aggregate_performance(measurements)
    benchmark = make_benchmark()
    cases = [BenchmarkCase(id="case-001", input="Great", expected="positive")]
    timestamp = datetime(2026, 8, 14, 12, 30, tzinfo=UTC)

    result = BenchmarkResult(
        schema_version="1.0",
        started_at=timestamp,
        ended_at=timestamp,
        benchmark_spec=benchmark,
        cases=cases,
        measurements=measurements,
        quality_summaries=quality_summaries,
        performance_summaries=performance_summaries,
    )

    assert result.benchmark_spec == benchmark
    assert result.cases == cases
    assert result.measurements == measurements
    assert result.quality_summaries == quality_summaries
    assert result.performance_summaries == performance_summaries


def test_export_benchmark_result_round_trip(tmp_path: Path) -> None:
    measurement = make_measurement(
        "model-a",
        True,
        client_duration_ns=987_654_321,
        ollama_duration_ns=876_543_210,
        load_duration_ns=765_432_109,
        prompt_token_count=123,
        prompt_duration_ns=654_321_098,
        output_token_count=45,
        output_duration_ns=543_210_987,
    )
    measurement.raw_response["nested"] = {"raw": [1, "two", True]}
    benchmark = make_benchmark(repetitions=3)
    cases = [
        BenchmarkCase(
            id="case-001",
            input="Round-trip input",
            expected="positive",
        )
    ]
    started_at = datetime(2026, 8, 14, 12, 30, tzinfo=UTC)
    ended_at = datetime(2026, 8, 14, 12, 31, tzinfo=UTC)
    result = BenchmarkResult(
        schema_version="1.0",
        started_at=started_at,
        ended_at=ended_at,
        benchmark_spec=benchmark,
        cases=cases,
        measurements=[measurement],
        quality_summaries=aggregate_quality([measurement]),
        performance_summaries=aggregate_performance([measurement]),
    )

    output_path = export_benchmark_result(result, tmp_path)
    loaded = BenchmarkResult.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )

    assert output_path.name == "test-20260814T123100000000Z.json"
    assert loaded == result
    assert loaded.schema_version == "1.0"
    assert loaded.started_at == started_at
    assert loaded.ended_at == ended_at
    assert loaded.benchmark_spec == benchmark
    assert loaded.cases == cases
    assert loaded.measurements[0].client_duration_ns == 987_654_321
    assert loaded.measurements[0].ollama_duration_ns == 876_543_210
    assert loaded.measurements[0].prompt_token_count == 123
    assert loaded.measurements[0].response_text == "positive"
    assert loaded.measurements[0].is_correct is True
    assert loaded.measurements[0].raw_response["nested"] == {"raw": [1, "two", True]}
