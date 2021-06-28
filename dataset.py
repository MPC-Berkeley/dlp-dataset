import json
import numpy as np

import yaml
from yaml.loader import SafeLoader

# Load parking map
with open('parking_map.yml') as f:
    MAP_DATA = yaml.load(f, Loader=SafeLoader)

ORIGIN = MAP_DATA['ORIGIN']
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

    def coords_from_utm(self, coords):
        """
        convert coordinates from utm to local
        coords: an array-like variable with length >=2, and the first two entry are x, y coordinates
        The function will only chaneg the first two, and keep the rest of entries unchanged.
        
        Return: a np-array
        """
        result = np.array(coords)
        result[0] = ORIGIN['x'] - coords[0]
        result[1] = ORIGIN['y'] - coords[1]
        return result

    def states_from_utm(self, inst_token):
        """
        convert states of an instance from utm coordinates to local
        """
        instance = self.get('instance', inst_token)
        # Offset the coordinates and heading
        transformed_states = {
            'coords': self.coords_from_utm(instance['coords']),
            'heading': instance['heading'] - np.pi,
            'speed': instance['speed'], 
            'acceleration': instance['acceleration'],
            'mode': instance['mode']
        }

        # Determine the sign of speed
        heading_vector = np.array([np.cos(transformed_states['heading']), 
                                   np.sin(transformed_states['heading'])])

        if instance['next']:
            next_inst = self.get('instance', instance['next'])
        else:
            next_inst = instance

        if instance['prev']:
            prev_inst = self.get('instance', instance['prev'])
        else:
            prev_inst = instance
        motion_vector = np.array(self.coords_from_utm(next_inst['coords'])) - np.array(self.coords_from_utm(prev_inst['coords']))

        if heading_vector @ motion_vector < 0:
            transformed_states['speed'] *= -1

        return transformed_states