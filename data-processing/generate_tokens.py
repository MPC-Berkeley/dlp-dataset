import argparse
import hashlib
import json
import ntpath
import os
import pandas as pd
import numpy as np

def gen_token(str):
    return hashlib.sha1(str.encode('utf-8')).hexdigest()

def write_frames():
    frames = {}
    frame_ids = np.unique(df['frame_id'].to_numpy()).astype(int)

    for idx in frame_ids:
        frame_df = df[df['frame_id'] == str(idx)]
        agent_ids = frame_df['id'].to_numpy().astype(int)
        frame = {
            'frame_token': gen_token('{}_frame_{}'.format(filename_base, idx)),
            'scene_token': scene_token,
            'timestamp': frame_df['timestamp'].iloc[0],
            'instances': [gen_token('{}_instance_{}_{}'.format(filename_base, idx, x)) for x in agent_ids],
            'prev_frame': '' if idx == 0 else gen_token('{}_frame_{}'.format(filename_base, idx - 1)),
            'next_frame': '' if idx == max(frame_ids) else gen_token('{}_frame_{}'.format(filename_base, idx + 1))
        }
        frames[frame['frame_token']] = frame

    with open(filename_stem + '_frames.json', 'w') as out:
        json.dump(frames, out)

def write_agents():
    agents = {}
    agent_ids = np.unique(df['id'].to_numpy())

    for idx in agent_ids:
        agent_df = df[df['id'] == idx]
        frame_ids = agent_df['frame_id'].to_numpy().astype(int)
        agent = {
            'agent_token': gen_token('{}_agent_{}'.format(filename_base, idx)),
            'scene_token': scene_token,
            'type': agent_df['type'].iloc[0],
            'size': [agent_df['length'].iloc[0], agent_df['width'].iloc[0]],
            'first_instance': gen_token('{}_instance_{}_{}'.format(filename_base, min(frame_ids), idx)),
            'last_instance': gen_token('{}_instance_{}_{}'.format(filename_base, max(frame_ids), idx))
        }
        agents[agent['agent_token']] = agent
    
    with open(filename_stem + '_agents.json', 'w') as out:
        json.dump(agents, out)

def write_instances():
    instances = {}

    for _, row in df.iterrows():
        frame_id, agent_id = row['frame_id'], row['id']
        agent_frame_ids = df[df['id'] == agent_id]['frame_id'].to_numpy().astype(int)
        instance = {
            'instance_token': gen_token('{}_instance_{}_{}'.format(filename_base, frame_id, agent_id)),
            'agent_token': gen_token('{}_agent_{}'.format(filename_base, agent_id)),
            'frame_token': gen_token('{}_frame_{}'.format(filename_base, frame_id)),
            'coords': [row['utm_x'], row['utm_y']],
            'heading': row['utm_angle'],
            'speed': row['speed'],
            'acceleration': [row['lateral_acceleration'], row['tangential_acceleration']],
            'mode': '',
            'prev': '' if frame_id == min(agent_frame_ids) else gen_token('{}_instance_{}_{}'.format(filename_base, frame_id - 1, agent_id)),
            'next': '' if frame_id == max(agent_frame_ids) else gen_token('{}_instance_{}_{}'.format(filename_base, frame_id - 1, agent_id))
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
        'first_frame': gen_token('{}_frame_{}'.format(filename_base, min(frame_ids))),
        'last_frame': gen_token('{}_frame_{}'.format(filename_base, max(frame_ids))),
        'agents': [gen_token('{}_agent_{}'.format(filename_base, idx)) for idx in agent_ids]
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
    scene_token = gen_token(filename_base)

    df = pd.read_csv(args.file)

    write_frames()
    write_agents()
    write_instances()
    write_scene()