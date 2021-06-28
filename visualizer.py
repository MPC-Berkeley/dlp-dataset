import numpy as np
from numpy.lib.type_check import imag
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from PIL import ImageDraw

from dataset import Dataset

import yaml
from yaml.loader import SafeLoader

# Load parking map
with open('parking_map.yml') as f:
    MAP_DATA = yaml.load(f, Loader=SafeLoader)

ORIGIN = MAP_DATA['ORIGIN']
MAP_SIZE = MAP_DATA['MAP_SIZE']
PARKING_AREAS = MAP_DATA['PARKING_AREAS']

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

    def plot_obstacles(self, ax, scene_token):
        """
        plot static obstacles in this scene
        """
        scene = self.dataset.get('scene', scene_token)
        for obstacle_token in scene['obstacles']:
            obstacle = self.dataset.get('obstacle', obstacle_token)
            corners = self._get_corners(self._from_utm(obstacle['coords']), obstacle['size'], obstacle['heading'])
            ax.add_patch(patches.Polygon(corners, linewidth=0))


    def plot_frame(self, frame_token):
        frame = self.dataset.get('frame', frame_token)
        fig, ax = plt.subplots()

        # Plot parking lines
        self.plot_lines(ax)

        # Plot static obstacles
        self.plot_obstacles(ax, frame['scene_token'])
        
        # Plot instances
        for inst_token in frame['instances']:
            instance = self.dataset.get('instance', inst_token)
            agent = self.dataset.get('agent', instance['agent_token'])
            if agent['type'] not in {'Pedestrian', 'Undefined'}:
                corners = self._get_corners(self._from_utm(instance['coords']), agent['size'], instance['heading'])
                ax.add_patch(patches.Polygon(corners, linewidth=0, fill=True, color='orange'))

        ax.set_aspect('equal')
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

        # Plot static obstacles
        self.plot_obstacles(ax, agent['scene_token'])
        
        # Plot the specified instance
        if agent['type'] not in {'Pedestrian', 'Undefined'}:
            corners = self._get_corners(self._from_utm(instance['coords']), agent['size'], instance['heading'])
            ax.add_patch(patches.Polygon(corners, linewidth=0, fill=True, color='red'))

        # Plot other instances
        frame = self.dataset.get('frame', instance['frame_token'])
        for _inst_token in frame['instances']:
            if _inst_token == inst_token:
                continue

            _instance = self.dataset.get('instance', _inst_token)
            _agent = self.dataset.get('agent', _instance['agent_token'])
            if _agent['type'] not in {'Pedestrian', 'Undefined'}:
                corners = self._get_corners(self._from_utm(_instance['coords']), _agent['size'], _instance['heading'])
                ax.add_patch(patches.Polygon(corners, linewidth=0, fill=True, color='orange'))
            
        ax.set_aspect('equal')
        ax.set_xlim(0, MAP_SIZE['x'])
        ax.set_ylim(0, MAP_SIZE['y'])
        plt.show()
    
class SemanticVisualizer(Visualizer):
    """
    Plot the frame as semantic images
    """
    def __init__(self, dataset, spot_margin=0.3, resolution=0.1):
        """
        instantiate the semantic visualizer
        """
        super().__init__(dataset)
        
        self.spot_margin = spot_margin

        self.res = resolution
        self.h = int(MAP_SIZE['y'] / self.res)
        self.w = int(MAP_SIZE['x'] / self.res)

        # Shrink the parking spaces a little bit
        for name in ['top_left_x', 'btm_left_x', 'btm_left_y', 'btm_right_y']:
            self.parking_spaces[name] -= self.spot_margin
        for name in ['top_right_x', 'btm_right_x', 'top_left_y', 'top_right_y']:
            self.parking_spaces[name] += self.spot_margin

    def plot_obstacles(self, scene_token):
        """
        plot static obstacles in this scene
        """
        # Base image
        img_array = np.zeros((self.h, self.w), dtype=np.uint8)
        img = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img)

        scene = self.dataset.get('scene', scene_token)
        for obstacle_token in scene['obstacles']:
            obstacle = self.dataset.get('obstacle', obstacle_token)
            corners_ground = self._get_corners(self._from_utm(obstacle['coords']), obstacle['size'], obstacle['heading'])
            corners_pixel = (corners_ground / self.res).astype('int32')

            draw.polygon([tuple(p) for p in corners_pixel], fill=255)

        return np.asarray(img)

    def plot_agents(self, frame_token):
        """
        plot all moving agents at this frame
        """
        # Base image
        img_array = np.zeros((self.h, self.w), dtype=np.uint8)
        img = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img)

        frame = self.dataset.get('frame', frame_token)
        # Plot instances
        for inst_token in frame['instances']:
            instance = self.dataset.get('instance', inst_token)
            agent = self.dataset.get('agent', instance['agent_token'])
            if agent['type'] not in {'Pedestrian', 'Undefined'}:
                corners_ground = self._get_corners(self._from_utm(instance['coords']), agent['size'], instance['heading'])
                corners_pixel = (corners_ground / self.res).astype('int32')

                draw.polygon([tuple(p) for p in corners_pixel], fill=255)

        return np.asarray(img)

    def spot_available(self, current_img_array, center, size):
        """
        detect whether a certain spot on the map is occupied or not by checking the pixel value
        current_img_array: the image array after plotting static obstacles and moving agents in channel 0 and 1
        center: center location (pixel) of the spot
        size: the size of the square window for occupancy detection

        return: True if empty, false if occupied
        """
        sum = 0
        for x in range(int(center[0]-size/2), int(center[0]+size/2)):
            for y in range(int(center[1]-size/2), int(center[1]+size/2)):
                sum += current_img_array[y, x, 0] + current_img_array[y, x, 1]
        
        return sum == 0

    def plot_spots(self, current_img_array):
        """
        plot empty spots
        """
        # Base image
        img_array = np.zeros((self.h, self.w), dtype=np.uint8)
        img = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img)

        for _, p in self.parking_spaces.iterrows():
            p_coords_ground = self._from_utm_list(p[2:10].to_numpy().reshape((4, 2)))
            p_coords_pixel = (np.array(p_coords_ground) / self.res).astype('int32')
            
            # Detect whether this spot is occupied or not
            # Only plot the spot if it is empty
            center = np.average(p_coords_pixel, axis=0).astype('int32')
            if self.spot_available(current_img_array, center, size=16):
                draw.polygon([tuple(p) for p in p_coords_pixel], fill=255)

        return np.asarray(img)

    def plot_frame(self, frame_token):
        """
        plot frame as a semantic image
        """
        img_array = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        frame = self.dataset.get('frame', frame_token)

        img_array[:, :, 0] = self.plot_agents(frame_token)
        img_array[:, :, 1] = self.plot_obstacles(frame['scene_token'])
        img_array[:, :, 2] = self.plot_spots(img_array)

        img = Image.fromarray(img_array, 'RGB').transpose(Image.FLIP_TOP_BOTTOM)

        img.show()