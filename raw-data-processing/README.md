# Creating JSON files from raw XML or CSV
(Internal Use) 

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