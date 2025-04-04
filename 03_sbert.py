#!/usr/bin/env python
# -*- coding: utf-8 -*-

from types import SimpleNamespace
import datasets
import numpy as np
import transformers
from datasets import DatasetDict
from sentence_transformers import (SentenceTransformer,
                                   SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments,
                                   losses, models)
from sentence_transformers.evaluation import (TripletEvaluator)
from sentence_transformers.training_args import BatchSamplers
from torch import nn
from transformers.utils import logging

from autobert.dataset import dataset_generate_pairs
from autobert.tokenizer import AssemblyTokenizer

transformers.AutoTokenizer.register("AssemblyTokenizer", AssemblyTokenizer)

# logging
logging.set_verbosity_info()
logger = logging.get_logger("transformers")


# config
config = SimpleNamespace(
    # training
    num_epochs=1,
    max_seq_length=1024,
    batch_size=8,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    # model
    embedding_dim=256,
    pooling_model="mean",
    device="cuda",
    # io
    model_load_path="bert-eklavya",
    model_save_path="sbert-eklavya",
)


# code
def main():
    word_embedding_model = models.Transformer(
        model_name_or_path=config.model_load_path,
        max_seq_length=config.max_seq_length,
    )
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    dense_model = models.Dense(
        in_features=pooling_model.get_sentence_embedding_dimension(),
        out_features=config.embedding_dim,
        activation_function=nn.Tanh(),
    )
    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model, dense_model], device=config.device
    )

    logger.info("Loading dataset...")
    ds = datasets.load_dataset("./EKLAVYA/data/")

    # drop lower 10% percentile of functions
    Q = np.percentile([len(f["inst_strings"]) for f in ds["train"]], q=10)
    logger.info(f"Q: {Q}")
    ds = ds.filter(lambda example: len(example["inst_strings"]) > Q)
    ds = ds.map(
        lambda example: {
            "text": "\n".join(example["inst_strings"]),
            "opcodes": hash(
                tuple([inst.split(" ")[0] for inst in example["inst_strings"]])
            ),
        }
    )

    logger.info("Generating pairs...")
    ds_sbert = DatasetDict({
        'train': dataset_generate_pairs(ds['train']),
        'test': dataset_generate_pairs(ds['test']),
    })

    logger.info("Dataset overview:")
    logger.info(ds_sbert)
    logger.info("Train examples: %d", len(ds_sbert['train']))
    logger.info("Test examples: %d", len(ds_sbert['test']))

    train_dataset = ds_sbert['train'].shuffle(seed=42)
    eval_dataset = ds_sbert["test"]
    loss = losses.TripletLoss(model=model)
    test_evaluator = TripletEvaluator(
        anchors=eval_dataset['anchor'],
        positives=eval_dataset['positive'],
        negatives=eval_dataset['negative'],
    )

    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=config.model_save_path, 
        # Optional training parameters:
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_total_limit=2,
        logging_steps=100,
    )

    logger.info("Training...")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=test_evaluator,
    )
    trainer.train()
    
    logger.info("Saving model...")
    model.save_pretrained(config.model_save_path)
    model.tokenizer.save_pretrained(config.model_save_path)
    logger.info("Model saved to %s", config.model_save_path)

    logger.info("Done!")

if __name__ == "__main__":
    main()
