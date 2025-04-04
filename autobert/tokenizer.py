#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import os
import re
from typing import Optional, Tuple
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import logging

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "model": 1024,
}

class AssemblyTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    _special_tokens = num_token, func_token = [AddedToken("<num>"), AddedToken("<func>")]

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        unk_token = "<unk>",
        sep_token = "<sep>",
        pad_token = "<pad>",
        cls_token = "<cls>",
        mask_token = "<mask>",
        additional_special_tokens = _special_tokens,
        **kwargs
    ):
        self.logger = logging.get_logger("transformers")
        self.do_lower_case = do_lower_case
        self.vocab = {}

        super().__init__(
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs
        )
        if os.path.isfile(vocab_file):
            self.vocab = self.load_vocab(vocab_file)
        else:
            self.logger.warning(f"{vocab_file} does not exist.")
            self.vocab = {t: id for id,t in enumerate(self.all_special_tokens)}

        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        # self.post_processor = TemplateProcessing(
        #     single="<cls> $A <sep>",
        #     pair="<cls> $A <sep> $B:1 <sep>:1",
        #     special_tokens=[
        #         ("<cls>", self.cls_token_id),
        #         ("<sep>", self.sep_token_id),
        #     ],
        # )

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text, **kwargs):
        self.logger.debug(f"_tokenize({self}, {text}, {kwargs})")
        if self.do_lower_case:
            text = text.lower()

        splits = re.findall(r'\b\w+\b|[*-\-\[\]\#]+', text)
        tokens = []

        for idx, token in enumerate(splits):
            if token.startswith("0x"):
                value = int(token, base=16)
                tokens.extend(str(value))
            elif token.isnumeric():
                value = int(token, base=0)
                tokens.extend(str(value))
            elif idx > 0 and splits[idx-1] == "call":
                tokens.append(str(self.func_token))
            else:
                tokens.append(token)

        return tokens

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0, token_ids_1 = None
    ):
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0, token_ids_1 = None, already_has_special_tokens = False
    ):
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    self.logger.warning(f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive.")
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)

    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        with open(vocab_file, "r", encoding="utf-8") as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.rstrip("\n")
            vocab[token] = index
        return vocab

    def build_vocab_from_iterator(self, iterable):
        vocab = {} # Ordered since Python 3.7

        for t in self.all_special_tokens:
            vocab[t] = vocab.get(t, 0) + 1

        for i in range(10):
            t = str(i)
            vocab[t] = vocab.get(t, 0) + 1

        for text in iterable:
            for t in self.tokenize(text):
                vocab[t] = vocab.get(t, 0) + 1

        self.vocab = {v: k for k,v in enumerate(vocab.keys())}
        return vocab
