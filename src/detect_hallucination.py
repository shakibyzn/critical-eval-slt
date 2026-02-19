import argparse
import asyncio
from pathlib import Path

from ollama import chat
from pydantic import BaseModel, Field

OUTPUT_FILE_NAME = "hallucinations_severity_temp.txt"


class ResponseTemplate(BaseModel):
    response: str = Field(
        ...,
        description=(
            "Severity level of hallucination. Possible values: "
            "No hallucination, "
            "Small hallucination, "
            "Partial hallucination, "
            "Full hallucination."
        ),
    )


def load_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SLT outputs for hallucinations."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the input file containing refs.txt and preds.txt",
    )
    return parser.parse_args()


async def evaluate_translation(ref_text: str, mt_text: str) -> str:
    prompt = f"""
System:

You are an expert evaluator of Sign Language Translation (SLT) outputs.

You will be given:
- A human reference translation
- A model predicted translation

Your task is to determine **how severe** the hallucination is in the predicted translation.

Definition:
A word in the translated text is considered a hallucination if it introduces information that is completely unrelated to the source text.

Assign one severity level according to these guidelines:

• No hallucination: The translated text does not contain any hallucinated words. 
• Small hallucination: The translated text contains 1-2 hallucinated words.  
• Partial hallucination: The translated text includes at least 3 hallucinated words, but not all words are hallucinated.  
• Full hallucination: Nearly all words in the translated text are hallucinated, with the exception of perhaps 1-2 words.

Note: The labels are mutually exclusive; for example, a translation with a partial hallucination does not qualify as a full hallucination.

User:
Reference Translation: {ref_text}
Predicted Translation: {mt_text}

Provide exactly one of the following labels as your response. Do not include any additional text or explanation:
• No hallucination
• Small hallucination
• Partial hallucination
• Full hallucination
"""

    messages = [{"role": "user", "content": prompt}]
    resp = chat(
        model="llama3:70b",
        messages=messages,
        format=ResponseTemplate.model_json_schema(),
        options={"temperature": 0},
    )
    validated_resp = ResponseTemplate.model_validate_json(resp.message.content)
    return validated_resp.response


async def main() -> None:
    args = load_args()
    output_file = Path(args.path) / OUTPUT_FILE_NAME
    with (
        open(Path(args.path) / "refs.txt") as ref_file,
        open(Path(args.path) / "preds.txt") as pred_file,
    ):
        references = ref_file.readlines()
        predictions = pred_file.readlines()

    with open(output_file, "w") as out_file:
        for src_text, mt_text in zip(references, predictions):
            src_text = src_text.strip()
            mt_text = mt_text.strip()
            result = await evaluate_translation(src_text, mt_text)
            out_file.write(result + "\n")

    print(f"Hallucination evaluation completed. Results saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
