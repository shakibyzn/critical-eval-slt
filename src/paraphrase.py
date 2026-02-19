import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

load_dotenv()

TRANSLATIONS_DIR = Path("translations")
OPENAI_MODEL = "gpt-4o-mini"
PROMPT = """Paraphrase the following sentence into a natural and fluent form. 
Do not alter any numbers written in words into digits or vice versa — keep the format as it is in the original text.
"""


class Sentence(BaseModel):
    paraphrased_sentence: str = Field(..., help="The paraphrased sentence.")


async def paraphrase_text(
    client: AsyncOpenAI, text: str, temperature: float = 0.0
) -> str:
    """Send one sentence for paraphrasing."""
    try:
        response = await client.responses.parse(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": text},
            ],
            text_format=Sentence,
            temperature=temperature,
        )
        return response.output_parsed.paraphrased_sentence
    except Exception as e:
        print(f"Error paraphrasing '{text}': {e}")
        raise ValueError(f"Paraphrasing failed for text: {text}")


async def paraphrase_text_multiple(
    client: AsyncOpenAI, text: str, n: int = 10, temperature: float = 0.7
) -> list[str]:
    """Generate `n` paraphrases of a sentence asynchronously."""
    tasks = []
    for _ in range(n):
        task = paraphrase_text(client, text, temperature=temperature)
        await asyncio.sleep(2)  # slight delay to avoid rate limits
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paraphrase sentences using OpenAI API."
    )
    parser.add_argument(
        "--modelname",
        type=str,
        choices=["signformer-s3d", "sem-slt", "spamo", "two-stream-slt"],
        default="signformer-s3d",
        help="The model name to determine input and output paths.",
    )
    parser.add_argument(
        "--use_multiref",
        action="store_true",
        help="Whether to use multiple references for paraphrasing.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    input_path = TRANSLATIONS_DIR / args.modelname / "preds.txt"
    output_path = TRANSLATIONS_DIR / args.modelname / "paraphrased.txt"
    if args.use_multiref:
        input_path = TRANSLATIONS_DIR / args.modelname / "refs.txt"
        output_path = TRANSLATIONS_DIR / args.modelname / "multiref_paraphrased.txt"

    with open(input_path, encoding="utf-8") as f:
        input_lines = f.readlines()

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    delay = 2  # initial delay

    # write to file
    with open(output_path, "a", encoding="utf-8") as f:
        for idx, line in enumerate(input_lines):
            if args.use_multiref:
                # For multiref
                paraphrased_variants = await paraphrase_text_multiple(
                    client, line.strip(), n=10, temperature=0.7
                )
                for variant in paraphrased_variants:
                    variant = variant[:-1] + " " + variant[-1]
                    f.write(variant.lower() + "\n")

                f.flush()

            else:
                paraphrased_text = await paraphrase_text(client, line.strip())
                # space before point
                if line.strip() == ".":
                    paraphrased_text = "."
                else:
                    paraphrased_text = (
                        paraphrased_text[:-1] + " " + paraphrased_text[-1]
                    )

                f.write(paraphrased_text.lower() + "\n")
                f.flush()

            print(
                f"Iter: {idx + 1}, Sleeping for {delay:.2f} seconds to avoid rate limits..."
            )

            # Apply delay every 5 iterations
            if (idx + 1) % 5 == 0:
                print(
                    f"Iter: {idx + 1}, Sleeping for {delay:.2f} seconds to avoid rate limits..."
                )
                await asyncio.sleep(delay)
                delay *= 1.12  # increase delay
                delay = min(delay, 8)

    print(f"Finished writing paraphrased output to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
