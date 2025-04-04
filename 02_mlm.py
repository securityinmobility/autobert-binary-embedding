#!/usr/bin/env python
# -*- coding: utf-8 -*-

from types import SimpleNamespace
import math
import datasets
import transformers
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers.utils import logging

from autobert.tokenizer import AssemblyTokenizer

transformers.AutoTokenizer.register("AssemblyTokenizer", AssemblyTokenizer)

# logging
logging.set_verbosity_info()
logger = logging.get_logger("transformers")

# config
config = SimpleNamespace(
    # training
    q = 13,
    num_epochs=10,
    max_seq_length=1024,
    batch_size=8,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    # model
    device="cuda",
    # io
    tokenizer = "./asm-tokenizer-multi/",
    model_save_path="bert-multi"
)


def main():
    logger.info("Loading tokenizer ...")
    tokenizer = AssemblyTokenizer.from_pretrained(config.tokenizer, local_files_only=True)
    tokenizer.model_max_length = config.max_seq_length

    logger.info("Loading datasets ...")
    ds = datasets.load_dataset("./EKLAVYA/data/")
    ds = ds.map(lambda f: {"text": "\n".join(f['inst_strings'])})
    preprocess_function = lambda f: tokenizer(f["text"], return_tensors="pt", return_special_tokens_mask=True, max_length=config.max_seq_length, truncation=True, padding=True)
    ds = ds.map(preprocess_function, batched=True)
    ds = ds.filter(lambda example: len(example['inst_strings']) > config.q)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    
    logger.info("Assembling model ...")
    model_config = transformers.BertConfig(
        vocab_size=tokenizer.vocab_size,
        num_attention_heads=12,
        max_position_embeddings=config.max_seq_length,
    )
    model = transformers.BertForMaskedLM(model_config).to(config.device)
    training_args = TrainingArguments(
        output_dir=config.model_save_path,
        overwrite_output_dir=True,
        save_strategy="epoch",
        eval_strategy="steps",
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=config.warmup_ratio,
        weight_decay=0.1,
        push_to_hub=False,
    )

    logger.info("Training ...")
    trainer = Trainer(
    model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=data_collator,
    )
    train_result = trainer.train()

    logger.info("Saving model ...")
    trainer.save_model()
    metrics = train_result.metrics
    
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.3}")
    print(eval_results)

if __name__ == '__main__':
    main()
