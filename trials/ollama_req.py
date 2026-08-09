import time
import httpx
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from statistics import median

URL = "http://127.0.0.1:11434/api/generate"

MODELS = [
    "qwen2.5:0.5b",
    "llama3.2:1b",
]

PROMPT = (
    "Classify the sentiment as positive, negative, or neutral. "
    "Return only the label. "
    "Text: im the dummest of all dumasses"
)


class ModelMeasurement(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )

    model_name: str = Field(min_length=1)
    repetition: int = Field(ge=1)
    response_text: str

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

        return self.prompt_token_count / (
            self.prompt_duration_ns / 1_000_000_000
        )


    @property
    def output_tokens_per_second(self) -> float | None:
        if self.output_duration_ns == 0:
            return None

        return self.output_token_count / (
            self.output_duration_ns / 1_000_000_000
        )
def run_model(
    client: httpx.Client,
    model: str,
    repetition: int,
) -> ModelMeasurement:
    request_body = {
        "model": model,
        "prompt": PROMPT,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 8,
            "num_ctx": 4096,
        },
    }

    started_ns = time.perf_counter_ns()
    response = client.post(URL, json=request_body, timeout=60.0)
    client_duration_ns = time.perf_counter_ns() - started_ns

    response.raise_for_status()
    result = response.json()

    prompt_duration_s = result["prompt_eval_duration"] / 1_000_000_000
    output_duration_s = result["eval_duration"] / 1_000_000_000

    prompt_tokens_per_second = (
        result["prompt_eval_count"] / prompt_duration_s
    )
    output_tokens_per_second = (
        result["eval_count"] / output_duration_s
    )

    return ModelMeasurement(
        model_name=model,
        repetition=repetition,
        response_text=result["response"],
        client_duration_ns=client_duration_ns,
        ollama_duration_ns=result["total_duration"],
        load_duration_ns=result["load_duration"],
        prompt_token_count=result["prompt_eval_count"],
        prompt_duration_ns=result["prompt_eval_duration"],
        output_token_count=result["eval_count"],
        output_duration_ns=result["eval_duration"],
        raw_response=result,
    )

def format_rate(rate: float | None) -> str:
    if rate is None:
        return "n/a"

    return f"{rate:.1f} tokens/s"


def print_measurement(measurement: ModelMeasurement) -> None:
    print(f"\nrepetition: {measurement.repetition}")
    print(f"model: {measurement.model_name}")
    print(f"response: {measurement.response_text.strip()}")
    print(
        f"client duration: "
        f"{measurement.client_duration_ns / 1_000_000_000:.3f} s"
    )
    print(
        f"Ollama duration: "
        f"{measurement.ollama_duration_ns / 1_000_000_000:.3f} s"
    )
    print(
        f"load duration: "
        f"{measurement.load_duration_ns / 1_000_000_000:.3f} s"
    )
    print(
        f"prompt speed: "
        f"{format_rate(measurement.prompt_tokens_per_second)}"
    )
    print(
        f"output speed: "
        f"{format_rate(measurement.output_tokens_per_second)}"
    )

def print_comparison(measurements: list[ModelMeasurement]) -> None:
    print("\ncomparison")

    for model in MODELS:
        samples = [
            measurement
            for measurement in measurements
            if measurement.model_name == model
        ]

        median_client_duration_ns = median(
            sample.client_duration_ns for sample in samples
        )
        median_prompt_speed = median(
            sample.prompt_tokens_per_second
            for sample in samples
            if sample.prompt_tokens_per_second is not None
        )
        median_output_speed = median(
            sample.output_tokens_per_second
            for sample in samples
            if sample.output_tokens_per_second is not None
        )

        print(f"\nmodel: {model}")
        print(f"samples: {len(samples)}")
        print(
            f"median client duration: "
            f"{median_client_duration_ns / 1_000_000_000:.3f} s"
        )
        print(f"median prompt speed: {median_prompt_speed:.1f} tokens/s")
        print(f"median output speed: {median_output_speed:.1f} tokens/s")

measurements: list[ModelMeasurement] = []

with httpx.Client(timeout=60.0) as client:
    health_response = client.get(
        "http://127.0.0.1:11434/api/version"
    )
    health_response.raise_for_status()

    for model in MODELS:
        for repetition in range(1, 4):
            measurement = run_model(
                client,
                model,
                repetition,
            )
            measurements.append(measurement)

print(f"\nrecorded measurements: {len(measurements)}")

for measurement in measurements:
    print_measurement(measurement)

print_comparison(measurements)

