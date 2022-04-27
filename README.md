# Dragon Lake Parking (DLP) Dataset API
![](https://img.shields.io/github/stars/MPC-Berkeley/dlp-dataset)
![](https://img.shields.io/github/license/MPC-Berkeley/dlp-dataset)
![](https://img.shields.io/badge/language-python-blue)

The API to work with the [Dragon Lake Parking (DLP) Dataset](https://sites.google.com/berkeley.edu/dlp-dataset)

Authors: Xu Shen (xu_shen@berkeley.edu), Michelle Pan, Vijay Govindarajan, Neelay Velingker, Alex Wong

Model Predictive Control (MPC) Lab at UC Berkeley

<div align=center>
<img height="400" src="https://github.com/MPC-Berkeley/dlp-dataset/blob/main/docs/dlp_vis.png"/>
<img height="300" src="https://github.com/MPC-Berkeley/dlp-dataset/blob/main/docs/dlp_semantic.png"/>  <img height="300" src="https://github.com/MPC-Berkeley/dlp-dataset/blob/main/docs/inst_centric.png"/>
</div>

## Install

1. Clone this repo
2. With your virtualenv activated, run `pip install -e .` in the root directory of this repo.
3. Place the JSON data files in the `./data` directory

## Usage in other projects

Import this dataset API as a package, e.g.

```
from dlp.dataset import Dataset as DlpDataset
```

## Quick-start tutorials

1. [`./notebooks/tutorial.ipynb`](https://github.com/MPC-Berkeley/dlp-dataset/blob/main/notebooks/tutorial.ipynb) explains the structure of the dataset and available APIs
2. [`./notebooks/visualization.ipynb`](https://github.com/MPC-Berkeley/dlp-dataset/blob/main/notebooks/visualization.ipynb) demonstrates the dataset by visualizing it with either matplotlib or PIL
