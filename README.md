# Dragon Lake Parking (DLP) Dataset API
The API to work with the [Dragon Lake Parking (DLP) Dataset](https://sites.google.com/berkeley.edu/dlp-dataset)

Authors: Xu Shen (xu_shen@berkeley.edu), Michelle Pan, Vijay Govindarajan, Neelay Velingker, Alex Wong

Model Predictive Control (MPC) Lab at UC Berkeley

![Normal Visualization](imgs/dlp_vis.png)

![Semantic Visualization](imgs/dlp_semantic.png)

![Instance Centric View](imgs/inst_centric.png)

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

1. `./notebooks/tutorial.ipynb` explains the structure of the dataset and available APIs
2. `./notebooks/visualization.ipynb` demonstrates the dataset by visualizing it with either matplotlib or PIL