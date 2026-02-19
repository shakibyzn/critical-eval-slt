"""
Automatic evaluation with BLEU, chrF++ and BLEURT for WMT-SLT 2022
Confidence intervals obtained via bootstrap resampling
Modified: now supports binning sentences by length
"""

import argparse
import os
import random
import re

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
        "--num_bins",
        type=int,
        default=0,
        help="Number of bins to group sentences by length (0 = no binning)",
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=False,
        help="Whether to save the evaluation results to a file",
    )
    return parser


def bin_sentences(references, predictions, num_bins, by="words"):
    """Split refs/preds into bins based on sentence length"""
    if by == "words":
        lengths = [len(s.split()) for s in references]
    else:
        lengths = [len(s) for s in references]

    max_len = max(lengths)
    bins = np.linspace(0, max_len, num_bins + 1, dtype=int)

    binned_refs, binned_preds = (
        [[] for _ in range(num_bins)],
        [[] for _ in range(num_bins)],
    )

    for ref, pred, l in zip(references, predictions, lengths):
        for i in range(num_bins):
            if bins[i] < l <= bins[i + 1]:
                binned_refs[i].append(ref)
                binned_preds[i].append(pred)
                break

    labels = [f"{bins[i] + 1}-{bins[i + 1]}" for i in range(num_bins)]
    return binned_refs, binned_preds, labels


def evaluate_bin(refs, preds, bleu, chrf, bleurt_scorer, bootstraps):
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

    referenceFile = os.path.join(args.iFolder, "refs.txt")
    predictionFile = os.path.join(args.iFolder, "preds.txt")

    if not os.path.exists(referenceFile) or not os.path.exists(predictionFile):
        raise FileNotFoundError("refs.txt or preds.txt not found in the input folder.")

    with open(referenceFile, "r") as f:
        reference = f.read().strip().split("\n")
    with open(predictionFile, "r") as f:
        hypothesis = f.read().strip().split("\n")

    bleu = BLEU(tokenize="none", force=True)
    chrf = CHRF(word_order=2)
    bleurt_scorer = score.BleurtScorer(args.BRTcheckpoint)

    if args.num_bins > 0:
        outputFile = os.path.join(args.iFolder, "evaluation_results_raw_bins.txt")
        binned_refs, binned_preds, labels = bin_sentences(
            reference, hypothesis, args.num_bins
        )
        print("# Bin, BLEU, chrF++, BLEURT, ROUGE-L")
        for i, (refs_bin, preds_bin, label) in enumerate(
            zip(binned_refs, binned_preds, labels)
        ):
            if not refs_bin:  # skip empty bins
                continue
            results = evaluate_bin(
                refs_bin, preds_bin, bleu, chrf, bleurt_scorer, args.bootstraps
            )
            print(f"{label}, " + ", ".join(results))
            if args.save:
                # write to csv file
                with open(outputFile, "a+") as f:
                    if i == 0:
                        f.write("Bin, BLEU, chrF++, BLEURT, ROUGE-L\n")
                    f.write(f"{label}, {', '.join(results)}\n")
    else:
        outputFile = os.path.join(args.iFolder, "evaluation_results_raw.txt")
        print("# All data, BLEU, chrF++, BLEURT, ROUGE-L")
        results = evaluate_bin(
            reference, hypothesis, bleu, chrf, bleurt_scorer, args.bootstraps
        )
        print("ALL, " + ", ".join(results))
        if args.save:
            with open(outputFile, "w") as f:
                f.write("BLEU, chrF++, BLEURT, ROUGE-L\n")
                f.write(", ".join(results) + "\n")
            print(f"Results saved to {outputFile}")


if __name__ == "__main__":
    main()
