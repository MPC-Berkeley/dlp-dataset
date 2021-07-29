import json
import numpy as np
import os

import yaml
from yaml.loader import SafeLoader

_ROOT = os.path.abspath(os.path.dirname(__file__))

# Load parking map
with open(_ROOT + '/parking_map.yml') as f:
    MAP_DATA = yaml.load(f, Loader=SafeLoader)

PARKING_AREAS = MAP_DATA['PARKING_AREAS']

class Dataset:

    def __init__(self):
        self.frames = {}
        self.agents = {}
        self.instances = {}
        self.scenes = {}
        self.obstacles = {}

    def load(self, filename):
        with open(filename + '_frames.json') as f:
            self.frames.update(json.load(f))
        with open(filename + '_agents.json') as f:
            self.agents.update(json.load(f))
        with open(filename + '_instances.json') as f:
            self.instances.update(json.load(f))
        with open(filename + '_obstacles.json') as f:
            self.obstacles.update(json.load(f))
        with open(filename + '_scene.json') as f:
            scene = json.load(f)
            self.scenes[scene['scene_token']] = scene

    def get(self, obj_type, token):
        assert obj_type in ['frame', 'agent', 'instance', 'obstacle', 'scene']

        if obj_type == 'frame':
            return self.frames[token]
        elif obj_type == 'agent':
            return self.agents[token]
        elif obj_type == 'instance':
            return self.instances[token]
        elif obj_type == 'obstacle':
            return self.obstacles[token]
        elif obj_type == 'scene':
            return self.scenes[token]
        
    def list_scenes(self):
        return list(self.scenes.keys())

    def get_agent_instances(self, agent_token):
        agent_instances = []
        next_instance = self.agents[agent_token]['first_instance']
        while next_instance:
            inst = self.instances[next_instance]
            agent_instances.append(inst)
            next_instance = inst['next']
        return agent_instances

    def get_agent_future(self, instance_token, timesteps=5):
        return self._get_timeline('instance', 'next', instance_token, timesteps)

    def get_agent_past(self, instance_token, timesteps=5):
        return self._get_timeline('instance', 'prev', instance_token, timesteps)

    def get_future_frames(self, frame_token, timesteps=5):
        return self._get_timeline('frame', 'next', frame_token, timesteps)
    
    def get_past_frames(self, frame_token, timesteps=5):
        return self._get_timeline('frame', 'prev', frame_token, timesteps)

    def _get_timeline(self, obj_type, direction, token, timesteps):
        if obj_type == 'frame':
            obj_dict = self.frames
        elif obj_type == 'instance':
            obj_dict = self.instances

        timeline = [obj_dict[token]]
        next_token = obj_dict[token][direction]
        for _ in range(timesteps):
            if not next_token:
                break
            next_obj = obj_dict[next_token]
            timeline.append(next_obj)
            next_token = next_obj[direction]

        if direction == 'prev':
            timeline.reverse()
            
        return timeline

    def signed_speed(self, inst_token):
        """
        determine the sign of the speed
        """
        instance = self.get('instance', inst_token)

        heading_vector = np.array([np.cos(instance['heading']), 
                                   np.sin(instance['heading'])])

        if instance['next']:
            next_inst = self.get('instance', instance['next'])
        else:
            next_inst = instance

        if instance['prev']:
            prev_inst = self.get('instance', instance['prev'])
        else:
            prev_inst = instance
        motion_vector = np.array(next_inst['coords']) - np.array(prev_inst['coords'])

        if heading_vector @ motion_vector > 0:
            return instance['speed']
        else:
            return - instance['speed']

    def get_future_traj(self, inst_token, static_thres=0.02):
        """
        get the future trajectory of this agent, starting from the current frame
        The static section at the begining and at the end will be truncated

        static_thres: the threshold to determine whether it is static
        
        Output: T x 4 numpy array. (x, y, heading, speed). T is the time steps
        """
        traj = []

        next_token = inst_token
        while next_token:
            instance = self.get('instance', next_token)
            signed_speed = self.signed_speed(next_token)
            traj.append(np.array([instance['coords'][0], instance['coords'][1], instance['heading'], signed_speed]))

            next_token = instance['next']

        last_idx = len(traj) - 1

        # Find the first non-static index
        idx_start = 0
        while idx_start < last_idx:
            if abs(traj[idx_start][3]) < static_thres:
                idx_start += 1
            else:
                break

        # Find the last non-static index
        idx_end = last_idx
        while idx_end > 0:
            if abs(traj[idx_end][3]) < static_thres:
                idx_end -= 1
            else:
                break

        if idx_end > idx_start:
            return np.array(traj[idx_start:idx_end])
        else:
            # If all indices are static, only return the current time step
            return traj[0].reshape((-1, 4))



        