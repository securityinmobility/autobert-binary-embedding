---
configs:
- config_name: default
  data_files:
  - split: train
    path:
    - "data/train/arm/*.jsonl"
    - "data/train/x86/*.jsonl"
  - split: test
    path:
    - "data/test/arm/*.jsonl"
    - "data/test/x86/*.jsonl"
- config_name: pairs
  data_files:
  - split: train
    path: "data/train/pairs.jsonl"
  - split: test
    path: "data/test/pairs.jsonl"
---

# Multi-ISAs basic block dataset (MISA)
The MISA dataset is available [here](https://github.com/zhangxiaochuan/MIRROR).

Download the files from the Google Drive link and extract them to this folder, 
such that the resulting folder structure looks like this:
```
MISA
├── data
│   ├── test
│   │   ├── arm  [5855 entries]
│   │   ├── pairs.jsonl
│   │   └── x86  [7737 entries]
│   └── train
│       ├── arm  [6029 entries]
│       ├── pairs.jsonl
│       └── x86  [7973 entries]
└── README.md
```

To generate the pairs, use the function `dataset_generate_pairs` in `autobert.dataset`
like this:

```python
import datasets
from autobert.dataset import dataset_generate_pairs

ds = datasets.load_dataset('MISA/')
train_pairs = dataset_generate_pairs(ds['train'])
[...]
```
