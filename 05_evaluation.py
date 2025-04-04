#!/usr/bin/env python
# -*- coding: utf-8 -*-

from types import SimpleNamespace
import datasets
import transformers
from tqdm import tqdm
from sentence_transformers import (
    SentenceTransformer,
)
from sentence_transformers.evaluation import TripletEvaluator, BinaryClassificationEvaluator, SimilarityFunction
from transformers.utils import logging

import numpy as np

from autobert.dataset import dataset_generate_pairs
from autobert.tokenizer import AssemblyTokenizer

tqdm.pandas()
transformers.AutoTokenizer.register("AssemblyTokenizer", AssemblyTokenizer)

# logging
logging.set_verbosity_info()
logger = logging.get_logger("transformers")

# config
config = SimpleNamespace(
    # dataset
    Q=13,
    seed=42,
    # io
    models = ["bert-x86-eklavya/", "bert-mixed-misa/", "minilm-l6-v2-mixed-misa/"],
    eklavya="./EKLAVYA/data/",
    misa="./MISA/"
)


def main():
    # load datasets
    ds = datasets.load_dataset(config.eklavya, split="test")
    ds = ds.filter(lambda f: len(f["inst_strings"]) > config.Q)
    ds = ds.map(lambda f: {"text": "\n".join(f['inst_strings'])})

    # load models
    eval_models = { model: SentenceTransformer(model) for model in config.models }
    for model in eval_models.values():
        model.eval()

    # generate pairs
    pairs = dataset_generate_pairs(ds)

    # Triplet Accuracy
    eval_dataset = pairs
    for model_name, model in eval_models.items():
        evaluator = TripletEvaluator(
            anchors=eval_dataset["anchor"],
            positives=eval_dataset["positive"],
            negatives=eval_dataset["negative"],
            main_similarity_function=SimilarityFunction.COSINE,
            name=f"EKLAVYA {model_name[:-1]}"
        )
        results = evaluator(model, output_path=f"evaluation/")
        print(f"{model_name=}: {results}")

    # ROC curve
    eval_dataset = datasets.concatenate_datasets([
        datasets.Dataset.from_dict({
            "sentence1": pairs['anchor'],
            "sentence2": pairs['positive'],
            "label": [1.0] * len(pairs),
        }),
        datasets.Dataset.from_dict({
            "sentence1": pairs['anchor'],
            "sentence2": pairs['negative'],
            "label": [0.0] * len(pairs),
        })
    ])
    for model_name, model in eval_models.items():
        evaluator = BinaryClassificationEvaluator(
            sentences1=eval_dataset["sentence1"],
            sentences2=eval_dataset["sentence2"],
            labels=eval_dataset["label"],
            name=f"EKLAVYA {model_name[:-1]}"
        )
        results = evaluator(model, output_path=f"evaluation/")
        print(f"{model_name=}: {results}")


if __name__ == "__main__":
    main()
