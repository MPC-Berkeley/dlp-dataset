import argparse
import hashlib
import json
import ntpath
import os
import subprocess
from tqdm import tqdm
import pandas as pd
import numpy as np

def gen_token(key):
    return hashlib.sha1(key.encode('utf-8')).hexdigest()

def write_frames():
    frames = {}
    frame_ids = np.unique(df['frame_id'].to_numpy()).astype(int)

    for frame_id in tqdm(frame_ids):
        frame_df = df[df['frame_id'] == frame_id]
        agent_ids = frame_df['id'].to_numpy().astype(int)
        frame = {
            'frame_token': gen_token('{}_frame_{}'.format(hash_base, frame_id)),
            'scene_token': scene_token,
            'timestamp': frame_df['timestamp'].iloc[0],
            'instances': [gen_token('{}_instance_{}_{}'.format(hash_base, frame_id, agent_id)) for agent_id in agent_ids],
            'prev_frame': '' if frame_id == 0 else gen_token('{}_frame_{}'.format(hash_base, frame_id - 1)),
            'next_frame': '' if frame_id == max(frame_ids) else gen_token('{}_frame_{}'.format(hash_base, frame_id + 1))
        }
        frames[frame['frame_token']] = frame

    with open(filename_stem + '_frames.json', 'w') as out:
        json.dump(frames, out)

def write_agents():
    agents = {}
    agent_ids = np.unique(df['id'].to_numpy())

    for agent_id in tqdm(agent_ids):
        agent_df = df[df['id'] == agent_id]
        frame_ids = agent_df['frame_id'].to_numpy().astype(int)
        agent = {
            'agent_token': gen_token('{}_agent_{}'.format(hash_base, agent_id)),
            'scene_token': scene_token,
            'type': agent_df['type'].iloc[0],
            'size': [agent_df['length'].iloc[0], agent_df['width'].iloc[0]],
            'first_instance': gen_token('{}_instance_{}_{}'.format(hash_base, min(frame_ids), agent_id)),
            'last_instance': gen_token('{}_instance_{}_{}'.format(hash_base, max(frame_ids), agent_id))
        }
        agents[agent['agent_token']] = agent
    
    with open(filename_stem + '_agents.json', 'w') as out:
        json.dump(agents, out)

def write_instances():
    instances = {}
    agent_ids = np.unique(df['id'].to_numpy())

    for agent_id in tqdm(agent_ids):
        agent_df = df[df['id'] == agent_id]
        frame_ids = agent_df['frame_id'].to_numpy().astype(int)

        for i, row in agent_df.iterrows():
            frame_id = row['frame_id']
            instance = {
                'instance_token': gen_token('{}_instance_{}_{}'.format(hash_base, frame_id, agent_id)),
                'agent_token': gen_token('{}_agent_{}'.format(hash_base, agent_id)),
                'frame_token': gen_token('{}_frame_{}'.format(hash_base, frame_id)),
                'coords': [row['utm_x'], row['utm_y']],
                'heading': row['utm_angle'],
                'speed': row['speed'],
                'acceleration': [row['lateral_acceleration'], row['tangential_acceleration']],
                'mode': '',
                'prev': '' if frame_id == min(frame_ids) else gen_token('{}_instance_{}_{}'.format(hash_base, frame_id - 1, agent_id)),
                'next': '' if frame_id == max(frame_ids) else gen_token('{}_instance_{}_{}'.format(hash_base, frame_id + 1, agent_id))
            }
            instances[instance['instance_token']] = instance

    with open(filename_stem + '_instances.json', 'w') as out:
        json.dump(instances, out)

def write_scene():
    frame_ids = np.unique(df['frame_id'].to_numpy()).astype(int)
    agent_ids = np.unique(df['id'].to_numpy())

    scene = {
        'scene_token': scene_token,
        'filename': filename_base,
        'timestamp': timestamp,
        'first_frame': gen_token('{}_frame_{}'.format(hash_base, min(frame_ids))),
        'last_frame': gen_token('{}_frame_{}'.format(hash_base, max(frame_ids))),
        'agents': [gen_token('{}_agent_{}'.format(hash_base, agent_id)) for agent_id in agent_ids]
    }

    with open(filename_stem + '_scene.json', 'w') as out:
        json.dump(scene, out)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='CSV file with scene data')
    parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')
    args = parser.parse_args()

    filename_stem = os.path.splitext(args.file)[0]
    filename_base = os.path.splitext(ntpath.basename(args.file))[0]

    process = subprocess.Popen(['mediainfo', filename_stem + '.MOV'], stdout=subprocess.PIPE)
    output, _ = process.communicate()
    timestamp = ' '.join(output.decode('utf-8').split('\n')[9].split()[-2:])
    hash_base = '{}_{}'.format(filename_base, timestamp)
    scene_token = gen_token(hash_base)

    df = pd.read_csv(args.file)

    print('writing frames')
    write_frames()

    print('writing agents')
    write_agents()

    print('writing instances')
    write_instances()

    print('writing scene')
    write_scene()