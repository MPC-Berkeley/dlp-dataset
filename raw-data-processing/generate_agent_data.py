from dlp.dataset import Dataset
from pathlib import Path
import pickle
import numpy as np

with open('spots_data.pickle', 'rb') as f:
    parking_spaces = pickle.load(f)

def closest_spot_to_coords(coords):
    return min(parking_spaces['parking_spaces'], key = lambda x: np.linalg.norm(coords - x))

if __name__ == "__main__":
    ds = Dataset()
    home_path = str(Path.home())
    ds.load(home_path + '/dlp-dataset/data/DJI_0012')
    all_scenes = ds.list_scenes()
    scene_token = all_scenes[0]
    scene = ds.get('scene', scene_token)
    i = -1
    agents = {}
    spot_dict = {}
    frame_dict = {}
    for agent_token in scene['agents']:
        agent_instances = ds.get_agent_instances(agent_token)
        agent = ds.get('agent', agent_token)
        if agent['type'] in {'Pedestrian', 'Undefined'}:
            continue
        size = agent['size']
        dv = []
        v = []
        heading = []
        coords = []
        closest_spot = []
        dist_to_closest_spot = []
        t = []
        for inst in agent_instances:
            coords.append(inst['coords'])
            heading.append(inst['heading'])
            v.append(inst['speed'])
            dv.append(inst['acceleration'])
            if inst['frame_token'] in frame_dict:
                t.append(frame_dict[inst['frame_token']])
            else:
                val = ds.get('frame', inst['frame_token'])['timestamp']
                t.append(val)
                frame_dict[inst['frame_token']] = val

            tup = tuple(inst['coords'])
            if tup in spot_dict:
                closest_spot.append(spot_dict[tup][0])
                dist_to_closest_spot.append(spot_dict[tup][1])
            else:
                spot = min(range(len(parking_spaces['parking_spaces'])), key = lambda x: np.linalg.norm(inst['coords'] - parking_spaces['parking_spaces'][x]))
                dist = np.linalg.norm(inst['coords'] - parking_spaces['parking_spaces'][spot])
                closest_spot.append(spot)
                dist_to_closest_spot.append(dist)
                spot_dict[tup] = (spot, dist)
        start_time = ds.get('frame', agent_instances[0]['frame_token'])['timestamp']
        time_diff = ds.get('frame', agent_instances[1]['frame_token'])['timestamp'] - start_time
        start = agent_instances[0]['coords']
        end = agent_instances[-1]['coords']
        agent = {}
        agent['start_time'] = start_time
        agent['start_spot']  = start
        agent['end_spot'] = end
        agent['vehicle_dim'] = size
        agent['v'] = v
        agent['dv_dt'] = dv
        agent['t'] = t
        agent['heading'] = heading
        agent['coords'] = coords
        agent['closest_spot'] = closest_spot
        agent['dist_to_closest_spot'] = dist_to_closest_spot
        agent['size'] = size
        """
        for start in range(len(v)):
            if v[start] != 0:
                break
        for end in range(len(v) - 1, -1, -1):
            if v[end] != 0:
                break
        agent['start_nonzero_v_timestep'] = start
        agent['end_nonzero_v_timestep'] = end
        agent['time_diff_timestep'] = time_diff
        """
        i += 1
        agents[i] = agent
    with open('agent_data.pickle', 'wb') as fp:
        pickle.dump(agents, fp)