import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Any, Literal, Protocol

from pydantic import Field, field_validator, model_validator

from themis.config import (
    BenchmarkCase,
    BenchmarkSpec,
    StrictModel,
    load_benchmark,
    load_dataset,
)
from themis.scoring import score_classification_response

OLLAMA_GENERATE_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_VERSION_URL = "http://127.0.0.1:11434/api/version"


# measures individual case for a model
class ModelMeasurement(StrictModel):
    model_name: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    repetition: int = Field(ge=1)
    response_text: str
    is_correct: bool

    client_duration_ns: int = Field(gt=0)
    ollama_duration_ns: int = Field(gt=0)
    load_duration_ns: int = Field(ge=0)

    prompt_token_count: int = Field(ge=0)
    prompt_duration_ns: int = Field(ge=0)
    output_token_count: int = Field(ge=0)
    output_duration_ns: int = Field(ge=0)

    raw_response: dict[str, Any]

    @property
    def prompt_tokens_per_second(self) -> float | None:
        if self.prompt_duration_ns == 0:
            return None
        return self.prompt_token_count / (self.prompt_duration_ns / 1_000_000_000)

    @property
    def output_tokens_per_second(self) -> float | None:
        if self.output_duration_ns == 0:
            return None
        return self.output_token_count / (self.output_duration_ns / 1_000_000_000)


# correctness
class ModelQualitySummary(StrictModel):
    model_name: str = Field(min_length=1)
    evaluated_run_count: int = Field(ge=0)
    correct_run_count: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)


class MetricStatistics(StrictModel):
    mean: float
    median: float


# efficiency
class ModelPerformanceSummary(StrictModel):
    model_name: str = Field(min_length=1)
    measured_run_count: int = Field(ge=0)
    client_duration_ns: MetricStatistics
    ollama_duration_ns: MetricStatistics
    load_duration_ns: MetricStatistics
    prompt_tokens_per_second: MetricStatistics | None
    output_tokens_per_second: MetricStatistics | None


# measures complete execution (all models, all cases, all repetitions)
class BenchmarkResult(StrictModel):
    schema_version: Literal["1.0"]
    started_at: datetime
    ended_at: datetime
    benchmark_spec: BenchmarkSpec
    cases: list[BenchmarkCase]
    measurements: list[ModelMeasurement]
    quality_summaries: list[ModelQualitySummary]
    performance_summaries: list[ModelPerformanceSummary]

    @field_validator("started_at", "ended_at")
    @classmethod
    def timestamps_are_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("benchmark timestamps must be timezone-aware UTC")
        return value

    @model_validator(mode="after")
    def timestamps_are_ordered(self) -> "BenchmarkResult":
        if self.ended_at < self.started_at:
            raise ValueError("benchmark end timestamp cannot precede start timestamp")
        return self


# protocol defines an interface by listing the methods an object must implement (smth like traits)
class Response(Protocol):
    def raise_for_status(self) -> None: ...

    def json(self) -> Any: ...


class HttpClient(Protocol):
    def post(self, url: str, *, json: dict[str, Any]) -> Response: ...


def _request_model(
    client: HttpClient,
    benchmark: BenchmarkSpec,
    case: BenchmarkCase,
    model_name: str,
) -> tuple[dict[str, Any], int]:
    request_body = {
        "model": model_name,
        "prompt": benchmark.prompt_template.format(input=case.input),
        "stream": False,
        "options": {
            "temperature": benchmark.generation.temperature,
            "num_predict": benchmark.generation.max_output_tokens,
            "num_ctx": benchmark.generation.context_length,
        },
    }

    started_ns = time.perf_counter_ns()
    response = client.post(OLLAMA_GENERATE_URL, json=request_body)
    client_duration_ns = max(1, time.perf_counter_ns() - started_ns)
    response.raise_for_status()
    return response.json(), client_duration_ns


def run_model(
    client: HttpClient,
    benchmark: BenchmarkSpec,
    case: BenchmarkCase,
    model_name: str,
    repetition: int,
) -> ModelMeasurement:
    result, client_duration_ns = _request_model(client, benchmark, case, model_name)
    response_text = result["response"]

    return ModelMeasurement(
        model_name=model_name,
        case_id=case.id,
        repetition=repetition,
        response_text=response_text,
        is_correct=score_classification_response(response_text, case.expected),
        client_duration_ns=client_duration_ns,
        ollama_duration_ns=result["total_duration"],
        load_duration_ns=result["load_duration"],
        prompt_token_count=result["prompt_eval_count"],
        prompt_duration_ns=result["prompt_eval_duration"],
        output_token_count=result["eval_count"],
        output_duration_ns=result["eval_duration"],
        raw_response=result,
    )


def execute_benchmark(
    client: HttpClient,
    benchmark: BenchmarkSpec,
    cases: list[BenchmarkCase],
) -> list[ModelMeasurement]:
    measurements: list[ModelMeasurement] = []

    for model_name in benchmark.models:
        if cases:
            for _ in range(benchmark.execution.warmup_runs):
                _request_model(client, benchmark, cases[0], model_name)

        for case in cases:
            for repetition in range(1, benchmark.execution.repetitions + 1):
                measurements.append(
                    run_model(client, benchmark, case, model_name, repetition)
                )

    return measurements


def aggregate_quality(
    measurements: list[ModelMeasurement],
) -> list[ModelQualitySummary]:
    if not measurements:
        return []

    grouped: dict[str, list[ModelMeasurement]] = defaultdict(list)
    for measurement in measurements:
        grouped[measurement.model_name].append(measurement)

    summaries: list[ModelQualitySummary] = []
    for model_name, model_measurements in grouped.items():
        evaluated_run_count = len(model_measurements)
        correct_run_count = sum(item.is_correct for item in model_measurements)
        summaries.append(
            ModelQualitySummary(
                model_name=model_name,
                evaluated_run_count=evaluated_run_count,
                correct_run_count=correct_run_count,
                accuracy=correct_run_count / evaluated_run_count,
            )
        )

    return summaries


def _metric_statistics(values: list[int | float]) -> MetricStatistics:
    return MetricStatistics(mean=mean(values), median=median(values))


def _optional_metric_statistics(
    values: list[float | None],
) -> MetricStatistics | None:
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return None
    return _metric_statistics(valid_values)


def aggregate_performance(
    measurements: list[ModelMeasurement],
) -> list[ModelPerformanceSummary]:
    if not measurements:
        return []

    grouped: dict[str, list[ModelMeasurement]] = defaultdict(list)
    for measurement in measurements:
        grouped[measurement.model_name].append(measurement)

    summaries: list[ModelPerformanceSummary] = []
    for model_name, model_measurements in grouped.items():
        summaries.append(
            ModelPerformanceSummary(
                model_name=model_name,
                measured_run_count=len(model_measurements),
                client_duration_ns=_metric_statistics(
                    [item.client_duration_ns for item in model_measurements]
                ),
                ollama_duration_ns=_metric_statistics(
                    [item.ollama_duration_ns for item in model_measurements]
                ),
                load_duration_ns=_metric_statistics(
                    [item.load_duration_ns for item in model_measurements]
                ),
                prompt_tokens_per_second=_optional_metric_statistics(
                    [item.prompt_tokens_per_second for item in model_measurements]
                ),
                output_tokens_per_second=_optional_metric_statistics(
                    [item.output_tokens_per_second for item in model_measurements]
                ),
            )
        )

    return summaries


def run_benchmark(
    client: HttpClient,
    benchmark_path: Path,
) -> BenchmarkResult:
    started_at = datetime.now(UTC)
    benchmark = load_benchmark(benchmark_path)
    cases = load_dataset(Path(benchmark.dataset))
    measurements = execute_benchmark(client, benchmark, cases)
    return BenchmarkResult(
        schema_version="1.0",
        started_at=started_at,
        ended_at=datetime.now(UTC),
        benchmark_spec=benchmark,
        cases=cases,
        measurements=measurements,
        quality_summaries=aggregate_quality(measurements),
        performance_summaries=aggregate_performance(measurements),
    )


def export_benchmark_result(
    result: BenchmarkResult,
    results_directory: Path = Path("results"),
) -> Path:
    results_directory.mkdir(parents=True, exist_ok=True)
    timestamp = result.ended_at.strftime("%Y%m%dT%H%M%S%fZ")
    output_path = results_directory / f"{result.benchmark_spec.name}-{timestamp}.json"
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return output_path
