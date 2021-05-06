# ParkingDataset
Parking Dataset by MPC Lab

### Creating JSON files

1. Download the XML and MOV files for a scene. They should have the same base filename (e.g. `DJI_0001.csv` and `DJI_0001.MOV`) and be in the same directory. The following assumes that they are in a subdirectory `data/`.

2. Convert the XML to a CSV.

    ```
    python data-processing/convert_xml.py data/DJI_0001.xml
    ```
    
3. Create JSON files.

    ```
    python data-processing/generate_tokens.py data/DJI_0001.csv
    ```
    
The files created will be:
- `data/DJI_0001_scene.json`
- `data/DJI_0001_frames.json`
- `data/DJI_0001_agents.json`
- `data/DJI_0001_instances.json`