# A Critical Study of Automatic Evaluation in Sign Language Translation - LREC 2026

Official implementation for the LREC 2026 paper: A Critical Study of Automatic Evaluation in Sign Language Translation

## Installation
```bash
# Make sure to install UV first.
uv venv .venv
uv pip install -r pyproject.toml
# Install and serve ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
# To evaluate using BLEURT
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
```

## Run
If you would like access to the Phoenix2014T translations produced by Signformer, SEM-SLT, SpaMo, or TwoStream-SLT models, please contact us at `shakibyzn <at> gmail.com`.  
Use **llama3:70b** for hallucination severity detection.
```bash
ollama pull llama3:70b
python src/detect_hallucination.py --path "your path containing refs.txt and preds.txt"
```

Use **gpt-4o-mini** for (multi-ref) paraphrasing.
```bash
python src/paraphrase.py --modelname sem-slt --use_multiref
```

To evaluate the output of SLT models with GEMBA or G-Eval, use one of the models from (**gpt-4.1-nano**, **qwen3:8b**, **llama3.1:8b**, **qwq:32b**).
```bash
ollama pull 'your-model-choice'
# with GEMBA Direct Assessment (DA)
python src/run.py --feature_name sem-slt --model_name llama3.1:8b --use_DA
# with GEMBA Direct Assessment (DA) and using paraphrased predictions
python src/run.py --feature_name sem-slt --model_name llama3.1:8b --use_DA --use_paraphs
# with G-Eval
python src/run.py --feature_name sem-slt --model_name llama3.1:8b
```

## Evaluation
Use one of the scripts from (eval.py, eval_multiref.py, or eval_hallucination_severity.py)
```bash
python evals/eval_hallucination_severity.py -i translations/sem-slt/
```

## Citation
If you find our work useful for your research, please cite:

```
@inproceedings{yazdani-etal-2026-critical,
  title = {A Critical Study of Automatic Evaluation in Sign Language Translation},
  author = {Yazdani, Shakib and HAMIDULLAH, Yasser and España-Bonet, Cristina and Avramidis, Eleftherios and Genabith, Josef van},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  month = {May},
  year = {2026},
  pages = {9535--9548},
  address = {Palma, Mallorca, Spain},
  publisher = {European Language Resources Association (ELRA)},
  editor = {Piperidis, Stelios and Bel, Núria and van den Heuvel, Henk and Ide, Nancy and Krek, Simon and Toral, Antonio},
  doi = {10.63317/4n2sooe4fb2i},
  abstract = {Automatic evaluation metrics are crucial for advancing sign language translation (SLT). Current SLT evaluation metrics, such as BLEU and ROUGE, are only text-based, and it remains unclear to what extent text-based metrics can reliably capture the quality of SLT outputs. To address this gap, we investigate the limitations of text-based SLT evaluation metrics by analyzing six metrics, including BLEU, chrF, and ROUGE, as well as BLEURT on the one hand, and large language model (LLM)-based evaluators such as G-Eval and GEMBA zero-shot direct assessment on the other hand. Specifically, we assess the consistency and robustness of these metrics under three controlled conditions: paraphrasing, hallucinations in model outputs, and variations in sentence length. Our analysis highlights the limitations of lexical overlap metrics and demonstrates that while LLM-based evaluators better capture semantic equivalence often missed by conventional metrics, they can also exhibit bias toward LLM-paraphrased translations. Moreover, although all metrics are able to detect hallucinations, BLEU tends to be overly sensitive, whereas BLEURT and LLM-based evaluators are comparatively lenient toward subtle cases. This motivates the need for multimodal evaluation frameworks that extend beyond text-based metrics to enable a more holistic assessment of SLT outputs.}
}
```

## Aknowledgements
We would like to sincerely thank the authors and contributors of [Signformer](https://github.com/EtaEnding/Signformer), [SEM-SLT](https://github.com/yhamidullah/sem-slt), [SpaMo](https://github.com/eddie-euijun-hwang/SpaMo), [TwoStream-SLT](https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork) for making their code publicly available.
