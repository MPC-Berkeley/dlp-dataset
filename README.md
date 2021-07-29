# Dragon Lake Parking (DLP) Dataset API
The API to work with the Dragon Lake Parking (DLP) Dataset

Authors: Xu Shen (xu_shen@berkeley.edu), Michelle Pan, Vijay Govindarajan

Model Predictive Control (MPC) Lab at UC Berkeley

## Install

1. Clone this repo
2. Run `pip install -e .` in the root directory of this repo
3. Place the JSON data files in the `./data` directory

## Usage in other projects

Import this dataset API as a package, e.g.

```
from dlp.dataset import Dataset as DlpDataset
```

## Quick-start tutorials

1. `./notebooks/tutorial.ipynb` explains the structure of the dataset
2. `./notebooks/visualization.ipynb` demonstrates the dataset by visualizing it with either matplotlib or PIL

## (Internal Use) Creating JSON files from raw XML or CSV

1. Download the XML and MOV files for a scene. They should have the same base filename (e.g. `DJI_0001.csv` and `DJI_0001.MOV`) and be in the same directory. The following assumes that they are in a subdirectory `data/`.

2. Convert the XML to a CSV.

    ```
    python raw-data-processing/convert_xml.py data/DJI_0001.xml
    ```
    
3. Create JSON files.

    Firstly make sure the `mediainfo` program is available. Otherwise install it with
    ```
    sudo apt install mediainfo
    ```

    For data files:

    ```
    python raw-data-processing/generate_tokens.py data/DJI_0001.csv
    ```
4. To create all json files with one-click:

    Firstly `cd` into folder `./raw-data-processing/`, then run the bash file 
    ```
    bash generate_all.sh
    ```

The files created will be:
- `data/DJI_0001_scene.json`
- `data/DJI_0001_frames.json`
- `data/DJI_0001_agents.json`
- `data/DJI_0001_instances.json`
- `data/DJI_0001_obstacles.json`