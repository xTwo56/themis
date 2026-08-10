import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


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


class ClassificationCase(StrictModel):
    id: str = Field(min_length=1)
    input: str = Field(min_length=1)
    expected: Literal["positive", "negative", "neutral"]


def load_benchmark(path: Path) -> BenchmarkSpec:
    with path.open(encoding="utf-8") as file:
        document: Any = yaml.safe_load(file)

    return BenchmarkSpec.model_validate(document)


def load_dataset(path: Path) -> list[ClassificationCase]:
    records: list[ClassificationRecord] = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            document: Any = json.loads(line)
            case = ClassificationCase.model_validate(document)
            records.append(case)

    return records
