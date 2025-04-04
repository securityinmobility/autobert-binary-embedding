#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import operator
import torch
import json
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from autobert.tokenizer import AssemblyTokenizer
from autobert.utils import find_first
import numpy as np
from datasets import Dataset
import os

class PairedDataset(Dataset):
    BATCH_SIZE = 1_000
    TRAIN_TEST_SPLIT = 0.2

    # name, start_address, end_address, instructions, project, architecture, compiler, optimization, source
    def __init__(self, path = "./dataset.pkl", **kwargs):
        self.functions = pd.read_pickle(path, **kwargs)
        fn_disassembly = operator.itemgetter("disassembly")
        self.functions["listing"] = self.functions.instructions.map(lambda lst: list(map(fn_disassembly, lst)))

        names_train, names_test = train_test_split(
            self.functions["name"].unique(),
            test_size=self.TRAIN_TEST_SPLIT,
            random_state=42
        )

    def __len__(self):
        return len(self.functions)

    def __getitem__(self, idx):
        return torch.tensor(self.tensors[i])

    def _stats(self):
        pass

    def batch_iterator():
        listings = self.df.instructions
        instructions = listings.map(lambda lst: list(map(operator.itemgetter("disassembly"), lst))).explode()

        for i in range(0, len(instructions), self.BATCH_SIZE):
            instructions = instructions.iloc[i : i + self.BATCH_SIZE]
            instructions = map(lambda s: s.lower(), instructions)
            yield list(instructions)


def find_first(haystack, needles):
    for needle in needles:
        if needle in haystack:
            return needle
    return None

def read_functions_from_file(path, db):
    filename = os.path.basename(path)


projects = ["zlib", "binutils", "libpng"]
architectures = ["aarch64", "x86"]
compilers = ["clang_9", "clang_10", "clang_11", "clang_12", "clang_13", "clang_14", "gcc_9", "gcc_10", "gcc_11"]
optimizations = ["O0", "O1", "O2", "O3"]

class AssemblyDataset(Dataset):
    def __init__(self, src_files):
        self.instructions = []
        self.metadata = []

        for src_file in src_files:
            print("🔥", src_file)
            filename = src_file.name
            metadata = {
                "project": find_first(filename, projects),
                "architecture": find_first(filename, architectures),
                "compiler": find_first(filename, compilers),
                "optimization": find_first(filename, optimizations),
            }
            instructions, metadata = zip(*[self._parse(f) for f in json.load(src_file.open("r"))])
            self.instructions.extend(instructions)
            self.metadata.extend(metadata)

    def _parse(self, function):
        """Extract data from json representation"""
        fn_disassembly = operator.itemgetter("disassembly")
        instructions = function.pop("instructions")
        return list(map(fn_disassembly, instructions)), function

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        return torch.tensor(self.instructions[idx])

    def __str__(self):
        return f"AssemblyDataset(n={len(self.instructions)})"


def dataset_generate_pairs(ds: Dataset, feature='text') -> Dataset:
    rng = np.random.default_rng(seed=42)
    df = ds.to_pandas()

    anchors, positives, negatives = [], [], []
    for anchor, row in df.iterrows():
        # Sample a positive with the same function name
        similars = list(df[df.loc[:, "name"] == row['name']].index)
        similars.remove(anchor)
        if len(similars) < 1:
            continue
        similar = df.loc[rng.choice(similars)]

        # Sample a negative at random, but make sure they are not equal
        while True:
            negative = df.loc[rng.choice(df.index)]
            if negative['opcodes'] != row['opcodes']:
                break
        anchors.append(row[feature])
        positives.append(similar[feature])
        negatives.append(negative[feature])
    return Dataset.from_dict({
        "anchor": anchors,
        "positive": positives,
        "negative": negatives
    })
