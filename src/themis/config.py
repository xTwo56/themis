import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator


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


class BenchmarkCase(StrictModel):
    id: str = Field(min_length=1)
    input: str = Field(min_length=1)
    expected: str = Field(min_length=1)


# we need a seprate class when we have a rule to enforce
# here the uniquesness of case_ids
class BenchmarkDataset(RootModel[list[BenchmarkCase]]):
    @model_validator(mode="after")
    def case_ids_are_unique(self) -> "BenchmarkDataset":
        case_ids = [case.id for case in self.root]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("benchmark case IDs must be unique")

        return self


def load_benchmark(path: Path) -> BenchmarkSpec:
    with path.open(encoding="utf-8") as file:
        document: Any = yaml.safe_load(file)

    return BenchmarkSpec.model_validate(document)


def load_dataset(path: Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            document: Any = json.loads(line)
            case = BenchmarkCase.model_validate(document)
            cases.append(case)

    return BenchmarkDataset.model_validate(cases).root
