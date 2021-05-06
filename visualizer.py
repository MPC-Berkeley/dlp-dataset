import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataset import Dataset

PARKING_AREAS = {
    'A': { # top
        'bounds': [
            [747035.85, 3856770.37],
            [746928.70, 3856774.70],
            [746928.57, 3856780.52],
            [747036.12, 3856776.37]],
        'areas': [
            {
                'shape': (1, 41),
                'coords': None
            }
        ]},
    'B': { # left, 1st down
        'bounds': [ 
            [747051.21, 3856783.27],
            [746987.48, 3856784.57],
            [746987.29, 3856795.55],
            [747056.37, 3856794.51],
            [747056.27, 3856788.74],
            [747055.37, 3856785.95],
            [747053.79, 3856784.27]],
        'areas': [
            {
                'shape': (2, 24),
                'coords': [
                    [747052.74, 3856783.14],
                    [746987.48, 3856784.57],
                    [746987.29, 3856795.55],
                    [747053.39, 3856794.66]
                ]
            }, {
                'shape': (1, 1),
                'coords': [
                    [747056.30, 3856788.76],
                    [747053.33, 3856788.88],
                    [747053.39, 3856794.66],
                    [747056.35, 3856794.56]
                ]
            }
        ]
    },
    'C': { # right, 1st down
        'bounds': [ 
            [746980.59, 3856784.77],
            [746928.45, 3856787.17],
            [746928.20, 3856797.68],
            [746980.27, 3856795.83]],
        'areas': [
            {
                'shape': (2, 21),
                'coords': None
            }
        ]},
    'D': { # left, 2nd down
        'bounds': [ 
            [747056.39, 3856801.82],
            [746987.16, 3856802.74],
            [746987.10, 3856814.05],
            [747056.59, 3856813.30]],
        'areas': [
            {
                'shape': (2, 25),
                'coords': None
            }
        ]},
    'E': { # right, 2nd down
        'bounds': [ 
            [746980.22, 3856802.86],
            [746928.10, 3856804.08],
            [746927.83, 3856814.88],
            [746980.08, 3856814.13]],
        'areas': [
            {
                'shape': (2, 21),
                'coords': None
            }
        ]},
    'F': { # left, 3rd down
        'bounds': [ 
            [747056.68, 3856820.66],
            [746987.12, 3856821.29],
            [746987.11, 3856832.48],
            [747057.00, 3856832.30]],
        'areas': [
            {
                'shape': (2, 25),
                'coords': None
            }
        ]},
    'G': { # right, 3rd down
        'bounds': [ 
            [746980.02, 3856821.24],
            [746927.69, 3856821.42],
            [746927.36, 3856832.17],
            [746980.07, 3856832.39]],
        'areas': [
            {
                'shape': (2, 21),
                'coords': None
            }
        ]},
    'H': { # left, 4th down
        'bounds': [ 
            [747057.01, 3856839.95],
            [746987.16, 3856839.50],
            [746987.20, 3856845.06],
            [747057.15, 3856845.88]],
        'areas': [
            {
                'shape': (1, 25),
                'coords': None
            }
        ]},
    'I': { # right, 4th down
        'bounds': [ 
            [746980.19, 3856839.42],
            [746929.63, 3856838.77],
            [746929.37, 3856844.29],
            [746980.16, 3856844.88]],
        'areas': [
            {
                'shape': (1, 20),
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
        return 747070 - coords[0], 3856850 - coords[1]

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

    def plot_frame(self, frame_token):
        frame = self.dataset.get('frame', frame_token)
        fig, ax = plt.subplots()

        for _, p in self.parking_spaces.iterrows():
            p_coords = self._from_utm_list(p[2:10].to_numpy().reshape((4, 2)))
            ax.add_patch(patches.Polygon(np.array(p_coords), ls='-', fill=False, color='#c7def0'))
            
        for inst_token in frame['instances']:
            instance = self.dataset.get('instance', inst_token)
            agent = self.dataset.get('agent', instance['agent_token'])
            if agent['type'] not in {'Pedestrian', 'Undefined'}:
                corners = self._get_corners(self._from_utm(instance['coords']), agent['size'], instance['heading'])
                ax.add_patch(patches.Polygon(corners, linewidth=0))

        ax.set_xlim(0, 145)
        ax.set_ylim(0, 85)
        plt.show()
    