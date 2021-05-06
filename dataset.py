import json

class Dataset:

    def __init__(self):
        self.frames = {}
        self.agents = {}
        self.instances = {}
        self.scenes = {}

    def load(self, filename):
        with open(filename + '_frames.json') as f:
            self.frames.update(json.load(f))
        with open(filename + '_agents.json') as f:
            self.agents.update(json.load(f))
        with open(filename + '_instances.json') as f:
            self.instances.update(json.load(f))
        with open(filename + '_scene.json') as f:
            scene = json.load(f)
            self.scenes[scene['scene_token']] = scene

    def get(self, obj_type, token):
        assert obj_type in ['frame', 'agent', 'instance', 'scene']

        if obj_type == 'frame':
            return self.frames[token]
        elif obj_type == 'agent':
            return self.agents[token]
        elif obj_type == 'instance':
            return self.instances[token]
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