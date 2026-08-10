from pathlib import Path

from themis.config import load_benchmark, load_dataset


BENCHMARK_PATH = Path("bench/sentiment.yaml")


def main() -> None:
    benchmark = load_benchmark(BENCHMARK_PATH)
    cases = load_dataset(Path(benchmark.dataset))

    print(f"benchmark: {benchmark.name}")
    print(f"models: {len(benchmark.models)}")
    print(f"dataset cases: {len(cases)}")

    for case in cases:
        print(f"{case.id}: expected={case.expected}")


if __name__ == "__main__":
    main()
