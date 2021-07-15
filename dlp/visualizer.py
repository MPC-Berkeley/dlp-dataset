import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from PIL import ImageDraw
import os

import yaml
from yaml.loader import SafeLoader

from dlp.dataset import Dataset

_ROOT = os.path.abspath(os.path.dirname(__file__))
# Load parking map
with open(_ROOT + '/parking_map.yml') as f:
    MAP_DATA = yaml.load(f, Loader=SafeLoader)

ORIGIN = MAP_DATA['ORIGIN']
MAP_SIZE = MAP_DATA['MAP_SIZE']
PARKING_AREAS = MAP_DATA['PARKING_AREAS']
WAYPOINTS = MAP_DATA['WAYPOINTS']

class Visualizer():

    def __init__(self, dataset: Dataset):
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

    def plot_waypoints(self, ax):
        """
        plot the waypoints on the map as scatters
        """
        for _, segment in WAYPOINTS.items():
            bounds = self._from_utm_list(segment['bounds'])
            points = np.linspace(bounds[0], bounds[1], num=segment['nums'], endpoint=True)
            ax.scatter(x=points[:, 0], y=points[:, 1], s=3, c='g')

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

        # Plot waypoints
        self.plot_waypoints(ax)

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

    def highlight_instance(self, inst_token):
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
    def __init__(self, dataset: Dataset, spot_margin=0.3, resolution=0.1, sensing_limit=20, steps=5, stride=5):
        """
        instantiate the semantic visualizer
        
        spot_margin: the margin for seperating spot rectangles
        resolution: distance (m) per pixel. resolution = 0.1 means 0.1m per pixel
        sensing_limit: the longest distance to sense along 4 directions (m). The side length of the square = 2*sensing_limit
        steps: the number history steps to plot. If no history is desired, set the steps = 0 and stride = any value.
        stride: the stride when getting the history. stride = 1 means plot the consecutive frames. stride = 2 means plot one in every 2 frames
        """
        super().__init__(dataset)
        
        self.spot_margin = spot_margin

        self.res = resolution
        self.h = int(MAP_SIZE['y'] / self.res)
        self.w = int(MAP_SIZE['x'] / self.res)

        self.sensing_limit = sensing_limit
        # 1/2 side length of the instance-centric crop. in pixel units.
        self.inst_ctr_size = int(self.sensing_limit / self.res)

        # Shrink the parking spaces a little bit
        for name in ['top_left_x', 'btm_left_x', 'btm_left_y', 'btm_right_y']:
            self.parking_spaces[name] -= self.spot_margin
        for name in ['top_right_x', 'btm_right_x', 'top_left_y', 'top_right_y']:
            self.parking_spaces[name] += self.spot_margin

        # Load the base map with drivable region
        self.base_map = Image.open(_ROOT + '/base_map.png').convert('RGB').resize((self.w, self.h)).transpose(Image.FLIP_TOP_BOTTOM)

        self.color = {'obstacle': (0, 0, 255),
                      'spot': (0, 255, 0),
                      'agent': (255, 255, 0),
                      'ego': (255, 0, 0)}

        self.steps = steps
        self.stride = stride

    def _color_transition(self, max_color, steps):
        """
        generate colors to plot the state history of agents

        max_color: 3-element tuple with r,g,b value. This is the color to plot the current state
        """
        # If we don't actually need the color band, return the max color directly
        if steps == 0:
            return [max_color]

        min_color = (int(max_color[0]/2), int(max_color[1]/2), int(max_color[2]/2))

        color_band = [min_color]

        for i in range(1, steps):
            r = int(min_color[0] + i * (max_color[0] - min_color[0]) / 2 / steps)
            g = int(min_color[1] + i * (max_color[1] - min_color[1]) / 2 / steps)
            b = int(min_color[2] + i * (max_color[2] - min_color[2]) / 2 / steps)
            color_band.append((r,g,b))

        color_band.append(max_color)

        return color_band

    def plot_obstacles(self, draw, fill, scene_token):
        """
        plot static obstacles in this scene
        """
        scene = self.dataset.get('scene', scene_token)

        for obstacle_token in scene['obstacles']:
            obstacle = self.dataset.get('obstacle', obstacle_token)
            corners_ground = self._get_corners(self._from_utm(obstacle['coords']), obstacle['size'], obstacle['heading'])
            corners_pixel = (corners_ground / self.res).astype('int32')

            draw.polygon([tuple(p) for p in corners_pixel], fill=fill)

    def plot_instance(self, draw, fill, instance):
        """
        plot a single instance at a single frame
        """
        agent = self.dataset.get('agent', instance['agent_token'])

        if agent['type'] not in {'Pedestrian', 'Undefined'}:
            corners_ground = self._get_corners(self._from_utm(instance['coords']), agent['size'], instance['heading'])
            corners_pixel = (corners_ground / self.res).astype('int32')

            draw.polygon([tuple(p) for p in corners_pixel], fill=fill)
    
    def plot_instance_timeline(self, draw, color_band, instance_timeline, stride):
        """
        plot the timeline of an instance
        """
        len_history = len(instance_timeline) - 1
        max_steps = np.floor( len_history / stride).astype(int)

        # History configuration
        for idx_step in range(max_steps, 0, -1):
            idx_history = len_history - idx_step * stride
            instance = instance_timeline[idx_history]
            self.plot_instance(draw=draw, fill=color_band[-1-idx_step], instance=instance)

        # Current configuration
        instance = instance_timeline[-1]
        self.plot_instance(draw=draw, fill=color_band[-1], instance=instance)

    def plot_agents(self, draw, fill, frame_token, steps, stride):
        """
        plot all moving agents and their history as fading rectangles
        """
        frame = self.dataset.get('frame', frame_token)

        color_band = self._color_transition(fill, steps)

        # Plot
        for inst_token in frame['instances']:
            instance_timeline = self.dataset.get_agent_past(inst_token, timesteps=steps*stride)
            self.plot_instance_timeline(draw, color_band, instance_timeline, stride)

    def spot_available(self, occupy_mask, center, size):
        """
        detect whether a certain spot on the map is occupied or not by checking the pixel value
        center: center location (pixel) of the spot
        size: the size of the square window for occupancy detection

        return: True if empty, false if occupied
        """
        sum = 0
        for x in range(center[0]-size, center[0]+size):
            for y in range(center[1]-size, center[1]+size):
                sum += occupy_mask.getpixel((x, y))
        
        return sum == 0

    def plot_spots(self, occupy_mask, draw, fill):
        """
        plot empty spots
        """
        for _, p in self.parking_spaces.iterrows():
            p_coords_ground = self._from_utm_list(p[2:10].to_numpy().reshape((4, 2)))
            p_coords_pixel = (np.array(p_coords_ground) / self.res).astype('int32')
            
            # Detect whether this spot is occupied or not
            # Only plot the spot if it is empty
            center = np.average(p_coords_pixel, axis=0).astype('int32')
            if self.spot_available(occupy_mask, center, size=8):
                draw.polygon([tuple(p) for p in p_coords_pixel], fill=fill)

    def plot_frame(self, frame_token):
        """
        plot frame as a semantic image
        """

        frame = self.dataset.get('frame', frame_token)

        # Create the binary mask for all moving objects on the map -- static obstacles and moving agents
        occupy_mask = Image.new(mode='1', size=(self.w, self.h))
        mask_draw = ImageDraw.Draw(occupy_mask)

        # Firstly register current obstacles and agents on the binary mask
        self.plot_obstacles(draw=mask_draw, fill=1, scene_token=frame['scene_token'])
        self.plot_agents(draw=mask_draw, fill=1, frame_token=frame_token, steps=0, stride=1)

        img_frame = self.base_map.copy()
        img_draw = ImageDraw.Draw(img_frame)

        # Then plot everything on the main img
        self.plot_spots(occupy_mask=occupy_mask, draw=img_draw, fill=self.color['spot'])
        self.plot_obstacles(draw=img_draw, fill=self.color['obstacle'], scene_token=frame['scene_token'])
        self.plot_agents(draw=img_draw, fill=self.color['agent'], frame_token=frame_token, steps=self.steps, stride=self.stride)

        return img_frame

    def inst_centric(self, img_frame, inst_token):
        """
        crop the local region around an instance and replot it in ego color. The ego instance is always pointing towards the west

        img_frame: the image of the SAME frame
        """
        img = img_frame.copy()
        draw = ImageDraw.Draw(img)

        # Replot this specific instance with the ego color
        color_band = self._color_transition(self.color['ego'], self.steps)

        instance_timeline = self.dataset.get_agent_past(inst_token, timesteps=self.steps*self.stride)
        self.plot_instance_timeline(draw, color_band, instance_timeline, self.stride)

        # The location of the instance in pixel coordinates, and the angle in degrees
        center = (self.dataset.states_from_utm(inst_token)['coords'] / self.res).astype('int32')
        angle_degree = self.dataset.states_from_utm(inst_token)['heading'] / np.pi * 180

        # Firstly crop a larger box which contains all rotations of the actual window
        outer_size = np.ceil(self.inst_ctr_size * np.sqrt(2))
        outer_crop_box = (center[0]-outer_size, center[1]-outer_size, center[0]+outer_size, center[1]+outer_size)
        
        # Rotate the larger box and crop the desired window size
        inner_crop_box = (outer_size-self.inst_ctr_size, outer_size-self.inst_ctr_size, outer_size+self.inst_ctr_size, outer_size+self.inst_ctr_size)

        img_instance = img.crop(outer_crop_box).rotate(angle_degree).crop(inner_crop_box)

        return img_instance
