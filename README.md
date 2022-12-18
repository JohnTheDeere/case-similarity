# case-similarity

## What?
This repo uses [Hugginface](https://huggingface.co/):hugs: and [dvc](https://dvc.org/) to create a pipeline for fine-tuning a sentence-embedding model using custom data in a structured way.

- Fine-tuning is done using question-aswering, e.g. embeddings of questions and corresponding answers are trained to be similar
- Currently uses [MPNet base v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) but easy to modify
- The pipeline uses dummy data here. For actual use modify the `create_data` step in the pipeline

## Setup

```
python -m venv env
source env/bin/activate
```

Upgrade pip and install requirements:

```
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Replace the `create_data` step with custom data. All you need is a dataset with questions and answers.
