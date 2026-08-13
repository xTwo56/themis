from pathlib import Path

import httpx

from themis.runner import (
    OLLAMA_VERSION_URL,
    MetricStatistics,
    ModelMeasurement,
    ModelPerformanceSummary,
    ModelQualitySummary,
    run_benchmark_file,
)


BENCHMARK_PATH = Path("bench/sentiment.yaml")


def format_rate(rate: float | None) -> str:
    if rate is None:
        return "n/a"
    return f"{rate:.1f} tokens/s"


def print_measurement(measurement: ModelMeasurement) -> None:
    print(f"\ncase: {measurement.case_id}")
    print(f"repetition: {measurement.repetition}")
    print(f"model: {measurement.model_name}")
    print(f"response: {measurement.response_text.strip()}")
    print(f"correct: {measurement.is_correct}")
    print(f"client duration: {measurement.client_duration_ns / 1e9:.3f} s")
    print(f"Ollama duration: {measurement.ollama_duration_ns / 1e9:.3f} s")
    print(f"load duration: {measurement.load_duration_ns / 1e9:.3f} s")
    print(f"prompt speed: {format_rate(measurement.prompt_tokens_per_second)}")
    print(f"output speed: {format_rate(measurement.output_tokens_per_second)}")


def print_quality_summary(summary: ModelQualitySummary) -> None:
    print(f"\nmodel: {summary.model_name}")
    print(f"evaluated runs: {summary.evaluated_run_count}")
    print(f"correct runs: {summary.correct_run_count}")
    print(f"accuracy: {summary.accuracy:.1%}")


def format_duration_statistics(statistics: MetricStatistics) -> str:
    return (
        f"mean={statistics.mean / 1e9:.3f} s, "
        f"median={statistics.median / 1e9:.3f} s"
    )


def format_rate_statistics(statistics: MetricStatistics | None) -> str:
    if statistics is None:
        return "n/a"
    return (
        f"mean={statistics.mean:.1f} tokens/s, "
        f"median={statistics.median:.1f} tokens/s"
    )


def print_performance_summary(summary: ModelPerformanceSummary) -> None:
    print(f"\nmodel: {summary.model_name}")
    print(f"measured runs: {summary.measured_run_count}")
    print(f"client duration: {format_duration_statistics(summary.client_duration_ns)}")
    print(f"Ollama duration: {format_duration_statistics(summary.ollama_duration_ns)}")
    print(f"load duration: {format_duration_statistics(summary.load_duration_ns)}")
    print(
        "prompt speed: "
        f"{format_rate_statistics(summary.prompt_tokens_per_second)}"
    )
    print(
        "output speed: "
        f"{format_rate_statistics(summary.output_tokens_per_second)}"
    )


def main() -> None:
    with httpx.Client(timeout=60.0) as client:
        health_response = client.get(OLLAMA_VERSION_URL)
        health_response.raise_for_status()
        result = run_benchmark_file(client, BENCHMARK_PATH)

    print(f"\nrecorded measurements: {len(result.measurements)}")
    for measurement in result.measurements:
        print_measurement(measurement)

    print("\nquality summary")
    if not result.quality_summaries:
        print("no evaluated runs")
    for summary in result.quality_summaries:
        print_quality_summary(summary)

    print("\nperformance summary")
    if not result.performance_summaries:
        print("no measured runs")
    for summary in result.performance_summaries:
        print_performance_summary(summary)


if __name__ == "__main__":
    main()
