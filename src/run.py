import argparse
import asyncio
from pathlib import Path

import pandas as pd

from scorer import Scorer

TRANSLATIONS_DIR = Path("translations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SLT translations")
    parser.add_argument(
        "--feature_name",
        type=str,
        default="ph14t_camgoz_signformer",
        help="Feature name",
    )
    parser.add_argument("--split", type=str, default="test", help="Data split")
    parser.add_argument("--model_name", type=str, default="qwq:32b", help="Model name")
    parser.add_argument(
        "--use_paraphs", action="store_true", help="Use paraphrased translations"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="instructor",
        help="Evaluation mode: instructor or openai",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode",
    )
    parser.add_argument(
        "--use_DA",
        action="store_true",
        help="Use GEMBA direct assessment (DA) scoring instead of G-Eval",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    output_feature_name = args.feature_name
    if args.use_paraphs:
        output_feature_name = "paraphrased_" + args.feature_name

    # Initialize the Scorer
    init_scorer = Scorer(
        feature_name=output_feature_name,
        split=args.split,
        mode=args.mode,
        model_name=args.model_name,
    )

    # Sample input data
    df_reference = pd.Series(
        Path(TRANSLATIONS_DIR / args.feature_name / "refs.txt")
        .read_text()
        .splitlines(),
        name="text",
    ).to_frame()

    if args.use_paraphs:
        df_prediction = pd.Series(
            Path(TRANSLATIONS_DIR / args.feature_name / "paraphrased.txt")
            .read_text()
            .splitlines(),
            name="paraphrased",
        ).to_frame()
    else:
        df_prediction = pd.Series(
            Path(TRANSLATIONS_DIR / args.feature_name / "preds.txt")
            .read_text()
            .splitlines(),
            name="text",
        ).to_frame()

    # evaluation
    # sample test
    if args.debug:
        random_indices: list[int] = list(range(6))
        sample_references = df_reference.loc[df_reference.index.isin(random_indices)][
            "text"
        ].tolist()
        if args.use_paraphs:
            sample_predictions = df_prediction.loc[
                df_prediction.index.isin(random_indices)
            ]["paraphrased"].tolist()
        else:
            sample_predictions = df_prediction.loc[
                df_prediction.index.isin(random_indices)
            ]["text"].tolist()

    # complete dataset
    else:
        sample_references = df_reference["text"].tolist()
        if args.use_paraphs:
            sample_predictions = df_prediction["paraphrased"].tolist()
        else:
            sample_predictions = df_prediction["text"].tolist()

    await init_scorer.process_batch(
        sample_predictions,
        sample_references,
        use_DA=args.use_DA,
        mode=args.mode,
    )


if __name__ == "__main__":
    asyncio.run(main())
