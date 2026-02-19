"""
Automatic evaluation with BLEU, chrF++ and BLEURT for multi-reference settings.
Confidence intervals obtained via bootstrap resampling.
Modified: now supports multiple references per prediction
"""

import argparse
import os
import random
import re

import numpy as np
from rouge_score import rouge_scorer
from sacrebleu import BLEU, CHRF
from tqdm import tqdm  # Added for progress bar on long computations

from bleurt import score


def compute_rougeL_f1_multiref(
    predictions, references_per_prediction, bootstraps, use_stemmer=True
):
    """
    Computes ROUGE-L F1 with multiple references.
    For each prediction, it calculates the ROUGE-L F1 score against all its references
    and takes the maximum score. Bootstrapping is then applied to these max scores.
    """
    scorer = rouge_scorer.RougeScorer(rouge_types=["rougeL"], use_stemmer=use_stemmer)
    max_f1_scores = []
    for pred, refs in zip(predictions, references_per_prediction):
        if not refs or all(not r.strip() for r in refs):
            max_f1_scores.append(0.0)
            continue

        individual_scores = [scorer.score(ref, pred)["rougeL"].fmeasure for ref in refs]
        max_f1_scores.append(max(individual_scores) * 100)  # Scale to 0-100

    # Apply bootstrapping on the distribution of maximum F1 scores
    mean, interval = ci_bs(max_f1_scores, bootstraps, 95)
    return f"{mean:.2f} ± {interval:.2f}"


def compute_bleurt_multiref(
    predictions, references_per_prediction, bleurt_scorer, bootstraps
):
    """
    Computes BLEURT with multiple references by finding the max score for each prediction.
    This avoids batching issues with the BLEURT library.
    """
    max_bleurt_scores = []
    print("Calculating BLEURT scores (this may take a while)...")
    for pred, refs in tqdm(
        zip(predictions, references_per_prediction), total=len(predictions)
    ):
        if not refs or all(not r.strip() for r in refs):
            max_bleurt_scores.append(
                0.0
            )  # Or a more appropriate value like a low score
            continue

        # Score one prediction against all its references and take the max
        # We repeat the candidate for each reference to match list lengths
        scores_for_one_pred = bleurt_scorer.score(
            references=refs, candidates=[pred] * len(refs)
        )
        max_bleurt_scores.append(max(scores_for_one_pred))

    # Apply bootstrapping on the distribution of maximum BLEURT scores
    mean, interval = ci_bs(max_bleurt_scores, bootstraps, 95)
    return f"{mean:.3f} ± {interval:.3f}"


def ci_bs(distribution, n, confLevel):
    """Computes mean and confidence interval using bootstrap resampling."""
    bsScores = np.zeros(n)
    size = len(distribution)
    random.seed(16)
    for i in range(n):
        bootstrapedSys = np.array(
            [distribution[random.randint(0, size - 1)] for _ in range(size)]
        )
        bsScores[i] = np.mean(bootstrapedSys, 0)
    mean = np.mean(bsScores, 0)
    confidenceInterval = np.percentile(
        bsScores, [(100 - confLevel) / 2, 100 - (100 - confLevel) / 2]
    )
    return (mean, mean - confidenceInterval[0])


def get_parser():
    """Sets up the argument parser."""
    parser = argparse.ArgumentParser(
        description="Automatic evaluation with BLEU, chrF++, BLEURT and ROUGE-L for multi-reference settings."
    )
    parser.add_argument(
        "-i",
        "--iFolder",
        type=str,
        default="../txt/",
        metavar="<inputFolder>",
        help="Folder with preds.txt and multiref_paraphrased.txt",
    )
    parser.add_argument(
        "-b",
        "--bootstraps",
        type=int,
        default=25,
        metavar="<#bootstraps>",
        help="Number of bootstrap samples (default: 1000)",
    )
    parser.add_argument(
        "-c",
        "--BRTcheckpoint",
        type=str,
        default="/netscratch/syazdani/Signformer/bleurt/BLEURT-20",
        metavar="<BLEURT checkpoint>",
        help="Path to BLEURT checkpoint",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save the evaluation results to a file"
    )
    return parser


def evaluate(
    transposed_refs, refs_per_pred, preds, bleu, chrf, bleurt_scorer, bootstraps
):
    """Runs the evaluation for all specified metrics."""
    results = []

    # BLEU
    # Note: sacrebleu's corpus_score requires the `num_bootstraps` argument name
    res_bleu = bleu.corpus_score(preds, transposed_refs, bootstraps)
    bleu_val = re.search(r"\(μ = (.*?)\)", str(res_bleu)).group(1)

    # CHRF++
    # Note: sacrebleu's corpus_score requires the `num_bootstraps` argument name
    res_chrf = chrf.corpus_score(preds, transposed_refs, bootstraps)
    chrf_val = re.search(r"\(μ = (.*?)\)", str(res_chrf)).group(1)

    # BLEURT (max over references) - UNCOMMENTED
    bleurt_val = compute_bleurt_multiref(
        preds, refs_per_pred, bleurt_scorer, bootstraps
    )

    # ROUGE-L (max over references) - UNCOMMENTED
    rouge_val = compute_rougeL_f1_multiref(preds, refs_per_pred, bootstraps)

    # Update the results list with the actual calculated values
    results.extend([bleu_val, chrf_val, bleurt_val, rouge_val])
    return results


def main(args=None):
    """Main function to read files, run evaluations, and print results."""
    parser = get_parser()
    args = parser.parse_args(args)

    multiRefFile = os.path.join(args.iFolder, "multiref_paraphrased.txt")
    originalRefFile = os.path.join(args.iFolder, "refs.txt")
    predictionFile = os.path.join(args.iFolder, "paraphrased.txt")

    if not os.path.exists(multiRefFile) or not os.path.exists(predictionFile):
        raise FileNotFoundError(
            f"Error: {multiRefFile} or {predictionFile} not found in the input folder."
        )

    num_paraphrases = 10
    references_per_prediction = []

    with open(predictionFile, "r", encoding="utf-8") as f:
        hypothesis = f.read().strip().split("\n")

    with open(multiRefFile, "r", encoding="utf-8") as f:
        all_references = [line.strip() for line in f]

    with open(originalRefFile, "r", encoding="utf-8") as f:
        original_references = [line.strip() for line in f]

    # Iterate through each original reference and its index
    for i, original_ref in enumerate(original_references):
        # Calculate the start and end index for the paraphrased slice
        start = i * num_paraphrases
        end = start + num_paraphrases

        # Get the corresponding 10 paraphrased references
        paraphrased_refs = all_references[start:end]

        # Create the new list: [original] + [10 paraphrases]
        combined_refs = [original_ref] + paraphrased_refs

        # Add this new 11-item list to our main list
        references_per_prediction.append(combined_refs)

    # --- Verification ---
    # Each inner list has 11 items: the original reference + 10 paraphrases.
    assert len(references_per_prediction) == len(hypothesis)
    assert len(references_per_prediction[0]) == num_paraphrases + 1

    # Transpose references for sacrebleu's required format
    transposed_refs = list(map(list, zip(*references_per_prediction)))

    # Initialize scorers
    bleu = BLEU(tokenize="none", force=True)
    chrf = CHRF(word_order=2)
    bleurt_scorer = score.BleurtScorer(args.BRTcheckpoint)

    # Run evaluation
    results = evaluate(
        transposed_refs=transposed_refs,
        refs_per_pred=references_per_prediction,
        preds=hypothesis,
        bleu=bleu,
        chrf=chrf,
        bleurt_scorer=bleurt_scorer,
        bootstraps=args.bootstraps,
    )

    header = "BLEU, chrF++, BLEURT, ROUGE-L"
    results_str = ", ".join(results)

    print("\n--- Evaluation Results ---")
    print(header)
    print(results_str)

    if args.save:
        outputFile = os.path.join(args.iFolder, "evaluation_results_multiref.txt")
        with open(outputFile, "w", encoding="utf-8") as f:
            f.write(header + "\n")
            f.write(results_str + "\n")
        print(f"\nResults saved to {outputFile}")


if __name__ == "__main__":
    main()
