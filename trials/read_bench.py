from pathlib import Path
from typing import Any, Literal

import yaml
import json
from pydantic import BaseModel, ConfigDict, Field

BENCHMARK_PATH = Path("bench/sentiment.yaml")

class StrictModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )

class GenerationConfig(StrictModel):
    temperature: float = Field(ge=0.0)
    max_output_tokens: int = Field(ge=1)
    context_length: int = Field(ge=1)

class ExecutionConfig(StrictModel):
    warmup_runs: int = Field(ge=0)
    repetitions: int = Field(ge=1)

class BenchmarkSpec(StrictModel):
    name: str = Field(min_length=1)
    models: list[str] = Field(min_length=2)
    prompt_template: str = Field(min_length=1)
    generation: GenerationConfig
    execution: ExecutionConfig
    dataset: str = Field(min_length=1)
    scorer: Literal["classification"]

class ClassificationRecord(StrictModel):
    id: str = Field(min_length=1)
    input: str = Field(min_length=1)
    expected: Literal["positive", "negative", "neutral"]

def main() -> None:
    with BENCHMARK_PATH.open(encoding="utf-8") as file:
        document: Any = yaml.safe_load(file)

    benchmark = BenchmarkSpec.model_validate(document)

    cases: list[ClassificationCase] = []

    dataset_path = Path(benchmark.dataset)

    with dataset_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            case_document: Any = json.loads(line)
            case = ClassificationRecord.model_validate(case_document)
            cases.append(case)

    print(f"benchmark: {benchmark.name}")
    print(f"models: {len(benchmark.models)}")
    print(f"dataset cases: {len(cases)}")

    for case in cases:
        print(f"{case.id}: expected={case.expected}")


if __name__ == "__main__":
    main()
