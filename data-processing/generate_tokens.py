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

def find_agents_obstacles(threshold=0.02):
    """
    distinguish the agents that has ever moved between the purely static ones. Treat the static cars as obstalces
    threshold: speed threshold
    """
    car_ids = np.unique(df['id'].to_numpy()).astype(int)

    agent_ids = []
    obstacle_ids = []

    for id in car_ids:
        car_trace = df[df['id'] == id]
        if np.max(car_trace['speed']) > threshold:
            agent_ids.append(id)
        else:
            obstacle_ids.append(id)

    return agent_ids, obstacle_ids

def get_mode(frame_id, agent_id, steps_to_check=50, threshold=0.02):
    """
    Determine the mode of an agent between 'parked' or 'moving' by looking at the speed profile

    steps_to_check: number of instances to check forwardly and backwardly. size = 50 <-> 2s forward and 2s backward
    threshold: the speed threshold for static / moving
    """
    agent_trace = df[df['id'] == agent_id]

    max_frame_id = np.max(agent_trace['frame_id']).astype(int)
    frames_to_check = np.arange(np.maximum(0, frame_id-steps_to_check), np.minimum(max_frame_id, frame_id+steps_to_check))

    speed_profile = agent_trace[agent_trace['frame_id'].isin(frames_to_check)]['speed']

    return 'moving' if np.max(speed_profile) > threshold else 'parked'


def write_frames():
    frames = {}
    frame_ids = np.unique(df['frame_id'].to_numpy()).astype(int)

    for frame_id in tqdm(frame_ids):
        frame_df = df[df['frame_id'] == frame_id]

        current_car_ids = frame_df['id'].to_numpy().astype(int) # All cars visible in the current frame
        current_agent_ids = np.intersect1d(current_car_ids, agent_ids) # All agents (non-static) in the current frame

        frame = {
            'frame_token': gen_token('{}_frame_{}'.format(hash_base, frame_id)),
            'scene_token': scene_token,
            'timestamp': frame_df['timestamp'].iloc[0],
            'prev': '' if frame_id == 0 else gen_token('{}_frame_{}'.format(hash_base, frame_id - 1)),
            'next': '' if frame_id == max(frame_ids) else gen_token('{}_frame_{}'.format(hash_base, frame_id + 1)),
            'instances': [gen_token('{}_instance_{}_{}'.format(hash_base, frame_id, agent_id)) for agent_id in current_agent_ids]
        }
        frames[frame['frame_token']] = frame

    with open(filename_stem + '_frames.json', 'w') as out:
        json.dump(frames, out)

def write_agents():
    agents = {}

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
                'mode': get_mode(frame_id, agent_id),
                'prev': '' if frame_id == min(frame_ids) else gen_token('{}_instance_{}_{}'.format(hash_base, frame_id - 1, agent_id)),
                'next': '' if frame_id == max(frame_ids) else gen_token('{}_instance_{}_{}'.format(hash_base, frame_id + 1, agent_id))
            }
            instances[instance['instance_token']] = instance

    with open(filename_stem + '_instances.json', 'w') as out:
        json.dump(instances, out)

def write_obstacles():
    """
    write obstalces into json
    """
    obstacles = {}

    for obstacle_id in tqdm(obstacle_ids):
        obstacle_trace = df[df['id'] == obstacle_id]

        obstacle = {
            'obstacle_token': gen_token('{}_obstacle_{}'.format(hash_base, obstacle_id)),
            'scene_token': scene_token,
            'type': obstacle_trace['type'].iloc[0],
            'size': [obstacle_trace['length'].iloc[0], obstacle_trace['width'].iloc[0]],
            'coords': [obstacle_trace['utm_x'].iloc[0], obstacle_trace['utm_y'].iloc[0]],
            'heading': obstacle_trace['utm_angle'].iloc[0],
        }
        obstacles[obstacle['obstacle_token']] = obstacle

    with open(filename_stem + '_obstacles.json', 'w') as out:
        json.dump(obstacles, out)

def write_scene():
    frame_ids = np.unique(df['frame_id'].to_numpy()).astype(int)

    scene = {
        'scene_token': scene_token,
        'filename': filename_base,
        'timestamp': timestamp,
        'first_frame': gen_token('{}_frame_{}'.format(hash_base, min(frame_ids))),
        'last_frame': gen_token('{}_frame_{}'.format(hash_base, max(frame_ids))),
        'agents': [gen_token('{}_agent_{}'.format(hash_base, agent_id)) for agent_id in agent_ids],
        'obstacles': [gen_token('{}_obstacle_{}'.format(hash_base, obstacle_id)) for obstacle_id in obstacle_ids]
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

    agent_ids, obstacle_ids = find_agents_obstacles()

    print('writing frames')
    write_frames()

    print('writing agents')
    write_agents()

    print('writing instances')
    write_instances()

    print('writing obstacles')
    write_obstacles()

    print('writing scene')
    write_scene()