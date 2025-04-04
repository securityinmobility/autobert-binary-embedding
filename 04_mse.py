#!/usr/bin/env python
# -*- coding: utf-8 -*-

from types import SimpleNamespace
import datasets
import transformers
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    losses,
    models,
)
from torch import nn
from transformers.utils import logging

from autobert.tokenizer import AssemblyTokenizer

transformers.AutoTokenizer.register("AssemblyTokenizer", AssemblyTokenizer)

# logging
logging.set_verbosity_info()
logger = logging.get_logger("transformers")


# config
config = SimpleNamespace(
    # training
    num_epochs=20,
    max_seq_length=1024,
    batch_size=8,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    # model
    model="bert-base-uncased",
    tokenizer="asm-tokenizer-multi/",
    embedding_dim=256,
    pooling_model="mean",
    device="cuda",
    # io
    teacher_model="./sbert-eklavya",
    student_model="./sbert-misa",
    dataset=("./MISA", "pairs"),
)


def main():
    logger.info("Loading models ...")
    teacher_model = SentenceTransformer(config.teacher_model, device=config.device)
    word_embedding_model = models.Transformer(config.model, tokenizer_name_or_path=config.tokenizer)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode=config.pooling_model,
    )
    dense_model = models.Dense(
        in_features=pooling_model.get_sentence_embedding_dimension(),
        out_features=config.embedding_dim,
        activation_function=nn.Tanh(),
    )
    student_model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model, dense_model]
    )

    logger.info("Loading dataset...")
    ds = datasets.load_dataset(*config.dataset)
    ds.select_columns(["anchor", "positive"])

    def compute_labels(batch):
        return {
            "label": teacher_model.encode(batch["anchor"])
        }
    ds = ds.map(compute_labels, batched=True)

    logger.info("Preparing training arguments...")
    train_loss = losses.MSELoss(model=student_model)
    trainer = SentenceTransformerTrainer(
        model=student_model,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        loss=train_loss,
    )

    logger.info("Running training...")
    trainer.train()

    logger.info("Saving model")
    student_model.save(config.student_model)


if __name__ == "__main__":
    main()
