# Dragon Lake Parking (DLP) Dataset API
![](https://img.shields.io/github/stars/MPC-Berkeley/dlp-dataset)
![](https://img.shields.io/github/license/MPC-Berkeley/dlp-dataset)
![](https://img.shields.io/badge/language-python-blue)

The API to work with the [Dragon Lake Parking (DLP) Dataset](https://sites.google.com/berkeley.edu/dlp-dataset).

The [Dragon Lake Parking (DLP) Dataset](https://sites.google.com/berkeley.edu/dlp-dataset) contains annotated video and data of vehicles, cyclists, and pedestrians inside a parking lot.

Abundant vehicle parking maneuvers and interactions are recorded. To the best of our knowledge, this is the first and largest public dataset designated for the parking scenario (up to April 2022), featuring high data accuracy and a rich variety of realistic human driving behavior. To download trial sample or request full access, please visit the dataset webpage for more infomation.

> **Note**: If you experience connectivity issue to the webpage above, [try this backup page](https://cutt.ly/dlp-notion).

Authors: Xu Shen (xu_shen@berkeley.edu), Michelle Pan, Vijay Govindarajan, Neelay Velingker, Alex Wong, Yibin Li

Model Predictive Control (MPC) Lab at UC Berkeley

<div align=center>
<img width="600" src="https://github.com/MPC-Berkeley/dlp-dataset/blob/main/docs/dlp_vis.png"/>
  
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

1. [`notebooks/tutorial.ipynb`](notebooks/tutorial.ipynb) explains the structure of the dataset and available APIs
2. [`notebooks/visualization.ipynb`](notebooks/visualization.ipynb) demonstrates the dataset by visualizing it with either matplotlib or PIL

## Partner
Source Trajectory and Bounding box data were annotated and gathered with DataFromSky [TrafficSurvey](https://datafromsky.com/trafficsurvey/) - an AI video analytics-based service for gathering advanced traffic data.
<div align=center>
<img height="60" src="https://user-images.githubusercontent.com/31999415/184691899-b398a790-03d8-404f-91b7-ed42d46f3a59.png"/>      <img height="60" src="https://user-images.githubusercontent.com/31999415/184692013-87175e25-b520-4929-b25a-8adeac612b08.jpeg"/>      <img height="60" src="https://datafromsky.com/wp-content/uploads/2015/03/datafromsky-logo.svg"/>
</div>
