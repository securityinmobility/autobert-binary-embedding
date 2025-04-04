#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click
import transformers
from transformers.utils import logging
from tqdm import tqdm
from autobert.tokenizer import AssemblyTokenizer

transformers.AutoTokenizer.register("AssemblyTokenizer", AssemblyTokenizer)

# logging
logging.set_verbosity_info()
logger = logging.get_logger("transformers")


def read_file(filename, encoding):
    with open(filename, "r", encoding=encoding) as fp:
        return fp.read()

@click.command()
@click.option("-c", "--corpus-path", required=True, type=click.Path(exists=True), multiple=True, help="Path(s) to corpus file(s)")
@click.option("-o", "--output-path", required=True, type=click.Path(), help="Output path")
@click.option("-e", "--encoding", type=str, default="utf-8", help="File encoding")
def main(corpus_path, output_path, encoding):
    tokenizer = AssemblyTokenizer("", corpus_path=corpus_path, output_path=output_path, encoding=encoding)
    tokenizer.build_vocab_from_iterator(map(lambda filename: read_file(filename, encoding), tqdm(corpus_path)))
    logger.info("VOCAB SIZE:", tokenizer.vocab_size)
    tokenizer.save_pretrained(output_path)

if __name__ == '__main__':
    main()
