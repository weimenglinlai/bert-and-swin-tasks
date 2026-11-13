# BERT and Swin Transformer Tasks

Offline BERT text tasks (classification, NER, sentiment) and Swin Transformer image classification.

## Structure

```
bert_tasks/01_bert_basic_test.py
bert_tasks/02_sentiment.py
bert_tasks/03_ner.py
bert_tasks/04_sentence_similarity.py
swin/inference_swin_t.py
swin/inference_swinv2_t.py
swin/swin_env/.gitignore
swin/swin_env/Scripts/Activate.ps1
swin/swin_env/Scripts/activate.bat
swin/swin_env/Scripts/deactivate.bat
swin/swin_env/pyvenv.cfg
```

## Setup

Install dependencies from `requirements.txt` when present:

```bash
pip install -r requirements.txt
```

Data files, model weights and generated outputs are not included; place them under the paths referenced by the configuration before running.
