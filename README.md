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
@article{yazdani2025criticalstudyautomaticevaluation,
      title={A Critical Study of Automatic Evaluation in Sign Language Translation}, 
      author={Shakib Yazdani and Yasser Hamidullah and Cristina España-Bonet and Eleftherios Avramidis and Josef van Genabith},
      year={2025},
      eprint={2510.25434},
      url={https://arxiv.org/abs/2510.25434}, 
}
```

## Aknowledgements
We would like to sincerely thank the authors and contributors of [Signformer](https://github.com/EtaEnding/Signformer), [SEM-SLT](https://github.com/yhamidullah/sem-slt), [SpaMo](https://github.com/eddie-euijun-hwang/SpaMo), [TwoStream-SLT](https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork) for making their code publicly available.