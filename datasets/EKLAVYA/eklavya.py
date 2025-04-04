# coding=utf-8
# TODO: MIT license ?

import gzip
import json
import pickle
import os
import random
import glob
import datasets
from sklearn.model_selection import train_test_split
import click


_DESCRIPTION = """\
The dataset available from this page is the collection of function type
signatures, which includes function banaries, number of arguments and types.
It is a good dataset for people who want to try learning techniques or
heuristic approaches in binary analysis while spending less effort on
collecting and preprocessing.
"""
_CITATION = """\
@inproceedings {203650,
    author = {Zheng Leong Chua and Shiqi Shen and Prateek Saxena and Zhenkai Liang},
    title = {Neural Nets Can Learn Function Type Signatures From Binaries},
    booktitle = {26th USENIX Security Symposium (USENIX Security 17)},
    year = {2017},
    isbn = {978-1-931971-40-9},
    address = {Vancouver, BC},
    pages = {99--116},
    url = {https://www.usenix.org/conference/usenixsecurity17/technical-sessions/presentation/chua},
    publisher = {USENIX Association},
    month = aug,
}
"""
_HOMEPAGE = "https://github.com/shensq04/EKLAVYA"

logger = datasets.logging.get_logger(__name__)

class EklavyaDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="x64",
            version=VERSION,
            description="x64 assembly code"
        )
    ]
    DEFAULT_CONFIG_NAME = "x64"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "name": datasets.Value("string"),
                    "binary_filename": datasets.Value("string"),
                    "num_args": datasets.Value("uint32"),
                    "args_type": datasets.Sequence(datasets.Value("string")),
                    "ret_type": datasets.Value("string"),
                    "inst_strings": datasets.Sequence(datasets.Value("string")),
                    "inst_bytes": datasets.Array2D(shape=(None, None), dtype="uint8"),
                    "boundaries": datasets.Array2D(shape=(1,2), dtype="uint64"),
                }
            ),
            supervised_keys = None,
            homepage = _HOMEPAGE,
            citation = _CITATION,
        )

    class BinaryEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, bytes):
                return obj.hex()
            # pass to base class default method
            return super().default(self, obj)

        @staticmethod
        def deserialize(d):
            key = 'bin_raw_bytes'
            if key in d and isinstance(d.get(key), str):
                d[key] = bytes.fromhex(d[key])
            return d

    def __init__(self, files=None, seed=42, **kwargs):
        self.functions = []
        self.index = {}


        if files is None:
            files = glob.glob("smalldata/json/x64/*.json")

        for file in files:
            open_fn = gzip.open if file.endswith('.gz') else open
            with open_fn(file, "r") as fp:
                document = json.load(fp, object_hook=EklavyaDataset.BinaryEncoder.deserialize)
                functions = document['functions']
                for key,value in functions.items():
                    value['binary_filename'] = document['binary_filename']
                    self.functions.append((key, value))
                    pos = len(self.functions) - 1
                    if key in self.index:
                        self.index[key].append(pos)
                    else:
                        self.index[key] = [pos]

        function_names = self.index.keys()
        self.train_functions, self.test_functions = train_test_split(list(function_names),
            test_size=0.2, random_state=seed)

        # self.train_pairs, self.train_labels = self.generate_pairs(train_functions)
        # self.test_pairs, self.test_labels  = self.generate_pairs(test_functions)

    def save(self, dir: str = "data", compress=True):
        for split_name, split_data in [("train", self.train_functions), ("test", self.test_functions)]:
            ext = ".jsonl" + (".gz" if compress else "")
            filename = os.path.join(dir, split_name + ext)
            logger.info(f"{filename}")
            open_fn = gzip.open if compress else open
            with open_fn(filename, "wt") as fp:
                for key in split_data:
                    for idx in self.index[key]:
                        name, func = self.functions[idx]
                        fp.write(json.dumps({"name": name, **func}) + "\n")

    def features(self, function):
        return function['inst_strings']

    def generate_pairs(self, functions):
        pairs = []
        labels = []
        for function in functions:
            for idx in self.index[function]:
                # anchor
                a_key, a_value = self.functions[idx]
                if not len(self.index[a_key]) > 1:
                    logger.info(f"{a_key}")
                    continue

                # positive
                choices = self.index[a_key].copy()
                choices.remove(idx)
                p_idx = random.choice(choices)
                p_key, p_value = self.functions[p_idx]

                # negative
                choices = functions.copy()
                choices.remove(function)
                n_function = random.choice(choices)
                n_idx = random.choice(self.index[n_function])
                _, n_value = self.functions[n_idx]

                pair = tuple(map(self.features, (a_value, p_value, n_value)))
                pairs.append(pair)
                labels.append((idx, p_idx, n_idx))
        return pairs, labels

    def __len__(self, dataset='train'):
        data = self.test_pairs if dataset == 'test' else self.train_pairs
        return len(data)

    def __getitem__(self, idx, dataset='train'):
        data = self.test_pairs if dataset == 'test' else self.train_pairs
        return data[idx]

    @staticmethod
    def convert_original(files):
        """
        Convert the EKLAVYA smalldata dataset from the original pickle format
        to a more robust, but slightly bigger json representation.

        Especially relevant for newer python versions, such as python3.
        """
        for file in files:
            logger.info(f"{file}")
            with open(file, "rb") as fp:
                binary = pickle.load(fp, encoding="latin1")
                key = 'bin_raw_bytes'
                binary[key] = bytes(binary[key], encoding="latin1")
            basename, ext = os.path.splitext(file)
            with gzip.open(basename + ".json.gz", "wt") as fp:
                json.dump(binary, fp, cls=EklavyaDataset.BinaryEncoder)
