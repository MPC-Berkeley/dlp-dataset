import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataset import Dataset

ORIGIN = {'x': 747064, 'y': 3856846}

MAP_SIZE = {'x': 140, 'y': 80}

PARKING_AREAS = {
    'A': { # top
        'bounds': [
            [747035.47, 3856772.27],
            [746928.48, 3856772.27],
            [746928.48, 3856777.49],
            [747035.47, 3856777.49]],
        'areas': [
            {
                'shape': (1, 41),
                'coords': None
            }
        ]},
    'B': { # left, 1st down
        'bounds': [ 
            [747056.29, 3856784.60],
            [746987.46, 3856784.60],
            [746987.46, 3856795.60],
            [747056.29, 3856795.60]],
        'areas': [
            {
                'shape': (2, 25),
                'coords': None
            }
        ]},
    'C': { # right, 1st down
        'bounds': [ 
            [746980.18, 3856784.60],
            [746928.48, 3856784.60],
            [746928.48, 3856795.60],
            [746980.18, 3856795.60]],
        'areas': [
            {
                'shape': (2, 21),
                'coords': None
            }
        ]},
    'D': { # left, 2nd down
        'bounds': [ 
            [747056.29, 3856802.76],
            [746987.46, 3856802.76],
            [746987.46, 3856814.07],
            [747056.29, 3856814.07]],
        'areas': [
            {
                'shape': (2, 25),
                'coords': None
            }
        ]},
    'E': { # right, 2nd down
        'bounds': [ 
            [746980.18, 3856802.76],
            [746928.48, 3856802.76],
            [746928.48, 3856814.07],
            [746980.18, 3856814.07]],
        'areas': [
            {
                'shape': (2, 21),
                'coords': None
            }
        ]},
    'F': { # left, 3rd down
        'bounds': [ 
            [747056.29, 3856821.32],
            [746987.46, 3856821.32],
            [746987.46, 3856832.49],
            [747056.29, 3856832.49]],
        'areas': [
            {
                'shape': (2, 25),
                'coords': None
            }
        ]},
    'G': { # right, 3rd down
        'bounds': [ 
            [746980.18, 3856821.32],
            [746928.48, 3856821.32],
            [746928.48, 3856832.49],
            [746980.18, 3856832.49]],
        'areas': [
            {
                'shape': (2, 21),
                'coords': None
            }
        ]},
    'H': { # left, 4th down
        'bounds': [ 
            [747056.29, 3856839.52],
            [746987.46, 3856839.52],
            [746987.46, 3856845.05],
            [747056.29, 3856845.05]],
        'areas': [
            {
                'shape': (1, 25),
                'coords': None
            }
        ]},
    'I': { # right, 4th down
        'bounds': [ 
            [746980.18, 3856839.52],
            [746928.48, 3856839.52],
            [746928.48, 3856845.05],
            [746980.18, 3856845.05]],
        'areas': [
            {
                'shape': (1, 21),
                'coords': None
            }
        ]}
}

class Visualizer():

    def __init__(self, dataset):
        self.dataset = dataset
        self.parking_spaces = self._gen_spaces()

        plt.rcParams['figure.dpi'] = 125
    
    def _gen_spaces(self):
        df = pd.DataFrame()
        idx = 0

        for ax, area in PARKING_AREAS.items():
            for a in area['areas']:
                df = df.append(self._divide_rect(a['coords'] if a['coords'] else area['bounds'], *a['shape'], idx, ax))
                idx += a['shape'][0] * a['shape'][1]

        df.columns = ['id', 'area', 'top_left_x', 'top_left_y', 'top_right_x', 'top_right_y', 'btm_right_x', 'btm_right_y', 'btm_left_x', 'btm_left_y']
        return df

    def _from_utm(self, coords):
        return ORIGIN['x'] - coords[0], ORIGIN['y'] - coords[1]

    def _from_utm_list(self, coords):
        return list(map(lambda c: list(self._from_utm(c)), coords))

    def _get_corners(self, center, dims, angle):
        length, width = dims
        offsets = np.array([[ 0.5, -0.5],
                            [ 0.5,  0.5],
                            [-0.5,  0.5],
                            [-0.5, -0.5]])
        offsets_scaled = offsets @ np.array([[length, 0], [0, width]])

        adj_angle = np.pi - angle
        c, s = np.cos(adj_angle), np.sin(adj_angle)
        rot_mat = np.array([[c, s], [-s, c]])
        offsets_rotated = rot_mat @ offsets_scaled.T

        c = np.array([*center])
        c_stacked = np.vstack((c, c, c, c))
        return offsets_rotated.T + c_stacked

    def _divide_rect(self, coords, rows, cols, start, area):
        left_x = np.linspace(coords[0][0], coords[3][0], rows + 1)
        left_y = np.linspace(coords[0][1], coords[3][1], rows + 1)

        right_x = np.linspace(coords[1][0], coords[2][0], rows + 1)
        right_y = np.linspace(coords[1][1], coords[2][1], rows + 1)

        points = np.zeros((rows + 1, cols + 1, 2))
        for i in range(rows + 1):
            x = np.linspace(left_x[i], right_x[i], cols + 1)
            y = np.linspace(left_y[i], right_y[i], cols + 1)
            points[i] = np.array(list(zip(x, y)))

        df = pd.DataFrame()
        idx = start

        for r in range(rows):
            for c in range(cols):
                df = df.append([[idx+1, area, *points[r][c], *points[r][c+1], *points[r+1][c+1], *points[r+1][c]]])
                idx += 1
                
        return df

    def plot_lines(self, ax):
        """
        plot parking lines
        """
        for _, p in self.parking_spaces.iterrows():
            p_coords = self._from_utm_list(p[2:10].to_numpy().reshape((4, 2)))
            ax.add_patch(patches.Polygon(np.array(p_coords), lw=0.5, ls='--', fill=False, color='#a0a0a0')) # c7def0


    def plot_frame(self, frame_token):
        frame = self.dataset.get('frame', frame_token)
        fig, ax = plt.subplots()

        # Plot parking lines
        self.plot_lines(ax)
        
        # Plot instances
        for inst_token in frame['instances']:
            instance = self.dataset.get('instance', inst_token)
            agent = self.dataset.get('agent', instance['agent_token'])
            if agent['type'] not in {'Pedestrian', 'Undefined'}:
                corners = self._get_corners(self._from_utm(instance['coords']), agent['size'], instance['heading'])
                ax.add_patch(patches.Polygon(corners, linewidth=0))

        ax.set_xlim(0, MAP_SIZE['x'])
        ax.set_ylim(0, MAP_SIZE['y'])
        plt.show()

    def plot_instance(self, inst_token):
        """
        emphasize a certain instance in a frame
        """
        instance = self.dataset.get('instance', inst_token)
        agent = self.dataset.get('agent', instance['agent_token'])

        print("The type of this instance is %s" % agent['type'])

        fig, ax = plt.subplots()

        # Plot parking lines
        self.plot_lines(ax)
        
        # Plot the specified instance
        if agent['type'] not in {'Pedestrian', 'Undefined'}:
            corners = self._get_corners(self._from_utm(instance['coords']), agent['size'], instance['heading'])
            ax.add_patch(patches.Polygon(corners, linewidth=0, fill=True, color='orange'))

        # Plot other instances
        frame = self.dataset.get('frame', instance['frame_token'])
        for _inst_token in frame['instances']:
            if _inst_token == inst_token:
                continue

            _instance = self.dataset.get('instance', _inst_token)
            _agent = self.dataset.get('agent', _instance['agent_token'])
            if _agent['type'] not in {'Pedestrian', 'Undefined'}:
                corners = self._get_corners(self._from_utm(_instance['coords']), _agent['size'], _instance['heading'])
                ax.add_patch(patches.Polygon(corners, linewidth=0))
            

        ax.set_xlim(0, MAP_SIZE['x'])
        ax.set_ylim(0, MAP_SIZE['y'])
        plt.show()
    
    def plot_agent_trace(self, agent_token):
        """
        plot the trace of an agent's state
        """
        pass