# Neural Nets Can Learn Function Type Signatures From Binaries
The EKLAVYA dataset is available [here](https://github.com/shensq04/EKLAVYA).

Download the `binary.tar.gz` and `pickles.tar.gz` (or `clean_pickles.tar.gz`), 
and unpack them to this directory. Afterwards, the pickle files need to be converted
to jsonl-files to be usable with the [huggingface datasets library](https://huggingface.co/docs/datasets).
For this, we provide the function `convert_original` in `eklavya.py`.

The resulting directory structure should look like this:
```
EKLAVYA
├── binary
│   ├── x64  [2584 entries]
│   └── x86  [2584 entries]
├── data
│   ├── test.jsonl.gz
│   └── train.jsonl.gz
├── eklavya.py
├── README.md
└── smalldata
    ├── pickles
    │   └── x64  [40 entries]
```

To generate the function pairs, use the function `dataset_generate_pairs` in 
`autobert.dataset` like this:

```python
import datasets
from autobert.dataset import dataset_generate_pairs

ds = datasets.load_dataset('EKLAVYA/')
train_pairs = dataset_generate_pairs(ds['train'])
[...]
```
