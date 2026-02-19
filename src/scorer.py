import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from ollama import chat
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

OUTPUT_DIR = Path("outputs_temp")


def select_prompt(use_DA: bool = False) -> Path:
    base_path = Path(__file__).parent / "prompts"
    filename = "ref-only-DA.txt" if use_DA else "ref-only.txt"
    return base_path / filename


def write_dict_to_json(filename: str, data_dict: Dict[str, Any]) -> None:
    if os.path.isfile(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                data: list[Dict[str, Any]] = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(data_dict)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


class Metric(BaseModel):
    adequacy: float = Field(..., description="Adequacy score")
    fluency: float = Field(..., description="Fluency score")
    reason: str = Field(..., description="Reason for the score")


class MetricDA(BaseModel):
    score: float = Field(
        ..., description="Direct assessment score within a range of (0-100)"
    )
    reason: str = Field(..., description="Reason for the score")


class Scorer:
    def __init__(
        self,
        feature_name: str,
        split: str,
        mode: str = "instructor",
        model_name: str | None = None,
    ):
        self.feature_name = feature_name
        self.split = split
        if mode == "instructor":
            self.model_name = model_name

        elif mode == "openai":
            load_dotenv()
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            self.client = AsyncOpenAI(api_key=self.openai_api_key)

        else:
            raise ValueError("mode must be 'instructor' or 'openai'")

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    async def process(self, pred: str, ref: str, use_DA: bool = False) -> None:

        prompt_fp = select_prompt(use_DA=use_DA)
        assert prompt_fp is not None, f"Prompt file not found for use_DA={use_DA}"

        prompt = open(prompt_fp).read()
        instance = {"reference": ref, "prediction": pred}

        cur_prompt = prompt.replace("{{Reference}}", ref).replace(
            "{{Translation}}", pred
        )
        messages = [{"role": "user", "content": cur_prompt}]

        response = await self.client.responses.parse(
            model="gpt-4.1-nano",
            input=messages,
            temperature=0.0,
            text_format=MetricDA if use_DA else Metric,
        )
        all_responses = response.output_parsed
        result = {**instance, **all_responses.dict()}

        if use_DA:
            out_path = (
                OUTPUT_DIR
                / f"geval_{self.feature_name}_{self.split}_gpt-4.1-nano_DA_True.json"
            )
        else:
            out_path = (
                OUTPUT_DIR / f"geval_{self.feature_name}_{self.split}_gpt-4.1-nano.json"
            )

        write_dict_to_json(out_path, result)

    # @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(5))
    def process_instructor(self, pred: str, ref: str, use_DA: bool = False) -> None:

        prompt_fp = select_prompt(use_DA=use_DA)
        assert prompt_fp is not None, (
            f"Prompt file not found for use_DA={use_DA} in instructor mode"
        )

        prompt = open(prompt_fp).read()
        instance = {"reference": ref, "prediction": pred}

        cur_prompt = prompt.replace("{{Reference}}", ref).replace(
            "{{Translation}}", pred
        )

        messages = [{"role": "user", "content": cur_prompt}]
        resp = chat(
            model=self.model_name,
            messages=messages,
            format=MetricDA.model_json_schema()
            if use_DA
            else Metric.model_json_schema(),
            options={"temperature": 0},
        )
        resp = (
            MetricDA.model_validate_json(resp.message.content)
            if use_DA
            else Metric.model_validate_json(resp.message.content)
        )
        result = {**instance, **resp.dict()}
        if use_DA:
            out_path = (
                OUTPUT_DIR
                / f"geval_{self.feature_name}_{self.split}_{self.model_name}_DA_True.json"
            )
        else:
            out_path = (
                OUTPUT_DIR
                / f"geval_{self.feature_name}_{self.split}_{self.model_name}.json"
            )

        write_dict_to_json(out_path, result)

    async def process_batch(
        self, gen: list[str], gts: list[str], use_DA: bool, mode: str = "instructor"
    ) -> None:
        delay, max_delay, count = 0.5, 5, 0
        # create output directory if it doesn't exist
        OUTPUT_DIR.mkdir(exist_ok=True)

        for pred, ref in zip(gen, gts):
            if count % 30 == 0:
                print(f"Processing iteration: {count + 1}")
                delay = min(delay * 1.2, max_delay)

            count += 1
            try:
                if mode == "instructor":
                    self.process_instructor(pred, ref, use_DA)
                elif mode == "openai":
                    await asyncio.wait_for(
                        self.process(pred, ref, use_DA),
                        timeout=30,
                    )

                else:
                    raise ValueError("mode must be 'instructor' or 'openai'")

            except Exception as e:
                print(f"Batch-level retry due to: {e}")

            # delay
            if mode != "instructor":
                await asyncio.sleep(delay)
