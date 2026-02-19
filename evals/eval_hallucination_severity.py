"""
Automatic evaluation with BLEU, chrF++ and BLEURT for WMT-SLT 2022
Confidence intervals obtained via bootstrap resampling
Modified: now supports hallucination severity
"""

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer, scoring
from sacrebleu import BLEU, CHRF

from bleurt import score


def compute_rougeL_f1(predictions, references, use_stemmer=True):
    scorer = rouge_scorer.RougeScorer(rouge_types=["rougeL"], use_stemmer=use_stemmer)
    scores = [scorer.score(ref, pred) for ref, pred in zip(references, predictions)]
    aggregator = scoring.BootstrapAggregator()
    for s in scores:
        aggregator.add_scores(s)
    result = aggregator.aggregate()["rougeL"]
    mean_f1 = result.mid.fmeasure * 100
    ci = (result.high.fmeasure - result.low.fmeasure) / 2 * 100
    return f"{mean_f1:.2f} ± {ci:.2f}"


def ci_bs(distribution, n, confLevel):
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--iFolder",
        type=str,
        default="../txt/",
        metavar="<inputFolder>",
        help="Folder with refs.txt and preds.txt",
    )
    parser.add_argument(
        "-b",
        "--bootstraps",
        type=int,
        default=25,
        metavar="<#bootstraps>",
        help="Number of bootstrap samples (default 25)",
    )
    parser.add_argument(
        "-c",
        "--BRTcheckpoint",
        type=str,
        default="/netscratch/syazdani/Signformer/bleurt/BLEURT-20",
        metavar="<BLEURT checkpoint>",
        help="BLEURT checkpoint",
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=False,
        help="Whether to save the evaluation results to a file",
    )
    return parser


def evaluate(refs, preds, bleu, chrf, bleurt_scorer, bootstraps):
    results = []

    # BLEU
    res = bleu.corpus_score(preds, [refs], bootstraps)
    bleu_val = re.search(r"\(μ = (.*?)\)", str(res)).group(1)

    # CHRF++
    res = chrf.corpus_score(preds, [refs], bootstraps)
    chrf_val = re.search(r"\(μ = (.*?)\)", str(res)).group(1)

    # BLEURT
    sentenceScores = bleurt_scorer.score(references=refs, candidates=preds)
    mean, interval = ci_bs(sentenceScores, bootstraps, 95)
    bleurt_val = f"{mean:.3f} ± {interval:.3f}"

    # ROUGE-L
    rouge_val = compute_rougeL_f1(preds, refs)

    results.extend([bleu_val, chrf_val, bleurt_val, rouge_val])
    return results


def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args)

    # 1. Setup paths
    input_dir = Path(args.iFolder)
    ref_file = input_dir / "refs.txt"
    pred_file = input_dir / "preds.txt"
    hallucination_file = input_dir / "hallucinations_severity.txt"
    output_file = input_dir / "evaluation_results_hallucination_severity.txt"

    if not all(p.exists() for p in [ref_file, pred_file, hallucination_file]):
        raise FileNotFoundError("One or more required input files are missing.")

    # 2. Load data
    with open(ref_file, "r", encoding="utf-8") as f:
        references = [line.strip() for line in f]
    with open(pred_file, "r", encoding="utf-8") as f:
        hypotheses = [line.strip() for line in f]
    with open(hallucination_file, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f]

    if len(references) != len(hypotheses) or len(hypotheses) != len(labels):
        raise ValueError(
            f"Length mismatch: refs({len(references)}), preds({len(hypotheses)}), labels({len(labels)})"
        )

    # 3. Group data by hallucination severity
    categories = [
        "Full hallucination",
        "Partial hallucination",
        "Small hallucination",
        "No hallucination",
    ]
    grouped_data = defaultdict(lambda: {"refs": [], "preds": []})

    for ref, hyp, label in zip(references, hypotheses, labels):
        grouped_data[label]["refs"].append(ref)
        grouped_data[label]["preds"].append(hyp)

    # 4. Initialize Scorer
    bleu = BLEU(tokenize="none", force=True)
    chrf = CHRF(word_order=2)
    bleurt_scorer = score.BleurtScorer(args.BRTcheckpoint)

    # 5. Evaluate and collect results
    print("# Hallucinations, BLEU, chrF++, BLEURT, ROUGE-L")
    output_lines = []

    for category in categories:
        data = grouped_data[category]

        # Guard against empty categories to avoid division by zero in scorers
        if not data["refs"]:
            result_str = "N/A, N/A, N/A, N/A"
        else:
            results = evaluate(
                data["refs"], data["preds"], bleu, chrf, bleurt_scorer, args.bootstraps
            )
            result_str = ", ".join(results)

        row = f"{category}, {result_str}"
        print(row)
        output_lines.append(row)

    # 6. Save results
    if args.save:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# Hallucinations, BLEU, chrF++, BLEURT, ROUGE-L\n")
            f.write("\n".join(output_lines) + "\n")


if __name__ == "__main__":
    main()
