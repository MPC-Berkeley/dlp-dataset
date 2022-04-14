from dlp.dataset import Dataset

from tqdm import tqdm
from pathlib import Path

statistics = {'scene': 0,
            'frame': 0,
            'agent': 0,
            'instance': 0}

for i in tqdm(range(1, 31)):
    # Load dataset
    ds = Dataset()

    home_path = str(Path.home())
    ds.load(home_path + '/dlp-dataset/data/DJI_' +  str(i).zfill(4) )

    statistics['scene'] += 1
    
    scene = ds.get('scene', ds.list_scenes()[0])
    
    statistics['agent'] += len(scene['agents'])

    frame = ds.get('frame', scene['first_frame'])
    statistics['frame'] += 1
    while frame['next']:
        frame = ds.get('frame', frame['next'])
        statistics['frame'] += 1
        statistics['instance'] += len(frame['instances'])


print(statistics)

