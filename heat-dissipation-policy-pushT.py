"""
Heat Dissipation Policy for PushT
===================================

Drop-in replacement for Diffusion Policy on the PushT task.
Uses the same dataset, environment, and evaluation protocol.

Instead of:
  Forward:  add Gaussian noise to action trajectories
  Reverse:  denoise with learned network

We do:
  Forward:  blur action trajectories via 1D heat equation (temporal smoothing)
  Reverse:  restore sharp, purposeful trajectories from blurred ones

Key differences to look for:
  1. Trajectory smoothness — heat policy should produce smoother actions
  2. Coarse-to-fine hierarchy — you can inspect the "plan" at intermediate steps
  3. Inference speed — fewer reverse steps may be needed
  4. Temporal consistency — actions should be more coherent over time

Run this notebook top-to-bottom. Cells 0-5 are identical to the original
Diffusion Policy notebook (env, dataset, helpers). Cell 6 onward is new.
"""

# ============================================================
# CELL 0: Install dependencies (same as original)
# ============================================================
# !pip3 install torch torchvision diffusers scikit-image scikit-video \
#   zarr numcodecs pygame pymunk gym shapely gdown &> /dev/null

# ============================================================
# CELL 1: Imports
# ============================================================
from typing import Tuple, Sequence, Dict, Union, Optional
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import zarr
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm

import gym
from gym import spaces
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
def vwrite(path, imgs):
    h, w = imgs[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
    for img in imgs:
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    writer.release()
import gdown
import os

# ============================================================
# CELL 2: PushT Environment (identical to original — paste from notebook)
# ============================================================
# >>> PASTE THE FULL PushTEnv CLASS FROM THE ORIGINAL NOTEBOOK HERE <<<
# (Cell 2 from diffusion_policy_state_pusht_demo.ipynb)
# It's ~600 lines. Not reproduced here to save space.
# The environment is unchanged between diffusion and heat policy.

#@markdown ### **Environment**
#@markdown Defines a PyMunk-based Push-T environment `PushTEnv`.
#@markdown
#@markdown **Goal**: push the gray T-block into the green area.
#@markdown
#@markdown Adapted from [Implicit Behavior Cloning](https://implicitbc.github.io/)


positive_y_is_up: bool = False
"""Make increasing values of y point upwards.

When True::

    y
    ^
    |      . (3, 3)
    |
    |   . (2, 2)
    |
    +------ > x

When False::

    +------ > x
    |
    |   . (2, 2)
    |
    |      . (3, 3)
    v
    y

"""
BASE       = os.path.dirname(os.path.abspath(__file__))
ZARR_PATH  = os.path.join(BASE, 'pusht_cchi_v7_replay.zarr.zip')

def to_pygame(p: Tuple[float, float], surface: pygame.Surface) -> Tuple[int, int]:
    """Convenience method to convert pymunk coordinates to pygame surface
    local coordinates.

    Note that in case positive_y_is_up is False, this function wont actually do
    anything except converting the point to integers.
    """
    if positive_y_is_up:
        return round(p[0]), surface.get_height() - round(p[1])
    else:
        return round(p[0]), round(p[1])


def light_color(color: SpaceDebugColor):
    color = np.minimum(1.2 * np.float32([color.r, color.g, color.b, color.a]), np.float32([255]))
    color = SpaceDebugColor(r=color[0], g=color[1], b=color[2], a=color[3])
    return color

class DrawOptions(pymunk.SpaceDebugDrawOptions):
    def __init__(self, surface: pygame.Surface) -> None:
        """Draw a pymunk.Space on a pygame.Surface object.

        Typical usage::

        >>> import pymunk
        >>> surface = pygame.Surface((10,10))
        >>> space = pymunk.Space()
        >>> options = pymunk.pygame_util.DrawOptions(surface)
        >>> space.debug_draw(options)

        You can control the color of a shape by setting shape.color to the color
        you want it drawn in::

        >>> c = pymunk.Circle(None, 10)
        >>> c.color = pygame.Color("pink")

        See pygame_util.demo.py for a full example

        Since pygame uses a coordiante system where y points down (in contrast
        to many other cases), you either have to make the physics simulation
        with Pymunk also behave in that way, or flip everything when you draw.

        The easiest is probably to just make the simulation behave the same
        way as Pygame does. In that way all coordinates used are in the same
        orientation and easy to reason about::

        >>> space = pymunk.Space()
        >>> space.gravity = (0, -1000)
        >>> body = pymunk.Body()
        >>> body.position = (0, 0) # will be positioned in the top left corner
        >>> space.debug_draw(options)

        To flip the drawing its possible to set the module property
        :py:data:`positive_y_is_up` to True. Then the pygame drawing will flip
        the simulation upside down before drawing::

        >>> positive_y_is_up = True
        >>> body = pymunk.Body()
        >>> body.position = (0, 0)
        >>> # Body will be position in bottom left corner

        :Parameters:
                surface : pygame.Surface
                    Surface that the objects will be drawn on
        """
        self.surface = surface
        super(DrawOptions, self).__init__()

    def draw_circle(
        self,
        pos: Vec2d,
        angle: float,
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p = to_pygame(pos, self.surface)

        pygame.draw.circle(self.surface, fill_color.as_int(), p, round(radius), 0)
        pygame.draw.circle(self.surface, light_color(fill_color).as_int(), p, round(radius-4), 0)

        circle_edge = pos + Vec2d(radius, 0).rotated(angle)
        p2 = to_pygame(circle_edge, self.surface)
        line_r = 2 if radius > 20 else 1
        # pygame.draw.lines(self.surface, outline_color.as_int(), False, [p, p2], line_r)

    def draw_segment(self, a: Vec2d, b: Vec2d, color: SpaceDebugColor) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)

        pygame.draw.aalines(self.surface, color.as_int(), False, [p1, p2])

    def draw_fat_segment(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)

        r = round(max(1, radius * 2))
        pygame.draw.lines(self.surface, fill_color.as_int(), False, [p1, p2], r)
        if r > 2:
            orthog = [abs(p2[1] - p1[1]), abs(p2[0] - p1[0])]
            if orthog[0] == 0 and orthog[1] == 0:
                return
            scale = radius / (orthog[0] * orthog[0] + orthog[1] * orthog[1]) ** 0.5
            orthog[0] = round(orthog[0] * scale)
            orthog[1] = round(orthog[1] * scale)
            points = [
                (p1[0] - orthog[0], p1[1] - orthog[1]),
                (p1[0] + orthog[0], p1[1] + orthog[1]),
                (p2[0] + orthog[0], p2[1] + orthog[1]),
                (p2[0] - orthog[0], p2[1] - orthog[1]),
            ]
            pygame.draw.polygon(self.surface, fill_color.as_int(), points)
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p1[0]), round(p1[1])),
                round(radius),
            )
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p2[0]), round(p2[1])),
                round(radius),
            )

    def draw_polygon(
        self,
        verts: Sequence[Tuple[float, float]],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        ps = [to_pygame(v, self.surface) for v in verts]
        ps += [ps[0]]

        radius = 2
        pygame.draw.polygon(self.surface, light_color(fill_color).as_int(), ps)

        if radius > 0:
            for i in range(len(verts)):
                a = verts[i]
                b = verts[(i + 1) % len(verts)]
                self.draw_fat_segment(a, b, radius, fill_color, fill_color)

    def draw_dot(
        self, size: float, pos: Tuple[float, float], color: SpaceDebugColor
    ) -> None:
        p = to_pygame(pos, self.surface)
        pygame.draw.circle(self.surface, color.as_int(), p, round(size), 0)


def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f'Unsupported shape type {type(shape)}')
    geom = sg.MultiPolygon(geoms)
    return geom

# env
class PushTEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self,
            legacy=False,
            block_cog=None, damping=None,
            render_action=True,
            render_size=96,
            reset_to_state=None
        ):
        self._seed = None
        self.seed()
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        # Local controller params.
        self.k_p, self.k_v = 100, 20    # PD control.z
        self.control_hz = self.metadata['video.frames_per_second']
        # legcay set_state for data compatiblity
        self.legacy = legacy

        # agent_pos, block_pos, block_angle
        self.observation_space = spaces.Box(
            low=np.array([0,0,0,0,0], dtype=np.float64),
            high=np.array([ws,ws,ws,ws,np.pi*2], dtype=np.float64),
            shape=(5,),
            dtype=np.float64
        )

        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([0,0], dtype=np.float64),
            high=np.array([ws,ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64
        )

        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None
        self.reset_to_state = reset_to_state

    def reset(self):
        seed = self._seed
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping

        # use legacy RandomState for compatiblity
        state = self.reset_to_state
        if state is None:
            rs = np.random.RandomState(seed=seed)
            state = np.array([
                rs.randint(50, 450), rs.randint(50, 450),
                rs.randint(100, 400), rs.randint(100, 400),
                rs.randn() * 2 * np.pi - np.pi
                ])
        self._set_state(state)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for i in range(n_steps):
                # Step PD control.
                # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                self.agent.velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)

        # compute reward
        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)

        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        reward = np.clip(coverage / self.success_threshold, 0, 1)
        done = coverage > self.success_threshold
        terminated = done
        truncated = done

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self, mode):
        return self._render_frame(mode)

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple('TeleopAgent', ['act'])
        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act
        return TeleopAgent(act)

    def _get_obs(self):
        obs = np.array(
            tuple(self.agent.position) \
            + tuple(self.block.position) \
            + (self.block.angle % (2 * np.pi),))
        return obs

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here dosn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            'pos_agent': np.array(self.agent.position),
            'vel_agent': np.array(self.agent.velocity),
            'block_pose': np.array(list(self.block.position) + [self.block.angle]),
            'goal_pose': self.goal_pose,
            'n_contacts': n_contact_points_per_step}
        return info

    def _render_frame(self, mode):

        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        # Draw goal pose.
        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            goal_points = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface) for v in shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        # Draw agent and block.
        self.space.debug_draw(draw_options)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # the clock is aleady ticked during in step for "human"


        img = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action:
            if self.render_action and (self.latest_action is not None):
                action = np.array(self.latest_action)
                coord = (action / 512 * 96).astype(np.int32)
                marker_size = int(8/96*self.render_size)
                thickness = int(1/96*self.render_size)
                cv2.drawMarker(img, coord,
                    color=(255,0,0), markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size, thickness=thickness)
        return img


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        pos_block = state[2:4]
        rot_block = state[4]
        self.agent.position = pos_agent
        # setting angle rotates with respect to center of mass
        # therefore will modify the geometric position
        # if not the same as CoM
        # therefore should be modified first.
        if self.legacy:
            # for compatiblity with legacy data
            self.block.position = pos_block
            self.block.angle = rot_block
        else:
            self.block.angle = rot_block
            self.block.position = pos_block

        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)

    def _set_state_local(self, state_local):
        agent_pos_local = state_local[:2]
        block_pose_local = state_local[2:]
        tf_img_obj = st.AffineTransform(
            translation=self.goal_pose[:2],
            rotation=self.goal_pose[2])
        tf_obj_new = st.AffineTransform(
            translation=block_pose_local[:2],
            rotation=block_pose_local[2]
        )
        tf_img_new = st.AffineTransform(
            matrix=tf_img_obj.params @ tf_obj_new.params
        )
        agent_pos_new = tf_img_new(agent_pos_local)
        new_state = np.array(
            list(agent_pos_new[0]) + list(tf_img_new.translation) \
                + [tf_img_new.rotation])
        self._set_state(new_state)
        return new_state

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()

        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2)
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone.
        self.agent = self.add_circle((256, 400), 15)
        self.block = self.add_tee((256, 300), 0)
        self.goal_color = pygame.Color('LightGreen')
        self.goal_pose = np.array([256,256,np.pi/4])  # x, y, theta (in radians)

        # Add collision handeling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = 0.95    # 95% coverage.

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color('LightGray')    # https://htmlcolorcodes.com/color-names
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('RoyalBlue')
        self.space.add(body, shape)
        return body

    def add_box(self, position, height, width):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        shape.color = pygame.Color('LightSlateGray')
        self.space.add(body, shape)
        return body

    def add_tee(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1
        length = 4
        vertices1 = [(-length*scale/2, scale),
                                 ( length*scale/2, scale),
                                 ( length*scale/2, 0),
                                 (-length*scale/2, 0)]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [(-scale/2, scale),
                                 (-scale/2, length*scale),
                                 ( scale/2, length*scale),
                                 ( scale/2, scale)]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body


# ============================================================
# CELL 3: Env Demo (identical to original)
# ============================================================
# >>> PASTE FROM ORIGINAL NOTEBOOK <<<

# from huggingface_hub.utils import IGNORE_GIT_FOLDER_PATTERNS
#@markdown ### **Env Demo**
#@markdown Standard Gym Env (0.21.0 API)

# 0. create env object
env = PushTEnv()

# 1. seed env for initial state.
# Seed 0-200 are used for the demonstration dataset.
env.seed(1000)

# 2. must reset before use
obs, IGNORE_GIT_FOLDER_PATTERNS = env.reset()

# 3. 2D positional action space [0,512]
action = env.action_space.sample()

# 4. Standard gym step method
obs, reward, terminated, truncated, info = env.step(action)

# prints and explains each dimension of the observation and action vectors
with np.printoptions(precision=4, suppress=True, threshold=5):
    print("Obs: ", repr(obs))
    print("Obs:        [agent_x,  agent_y,  block_x,  block_y,    block_angle]")
    print("Action: ", repr(action))
    print("Action:   [target_agent_x, target_agent_y]")

# ============================================================
# CELL 4: Dataset helpers (identical to original)
# ============================================================
# >>> PASTE create_sample_indices, sample_sequence, normalize/unnormalize,
#     PushTStateDataset FROM ORIGINAL NOTEBOOK <<<

#@markdown ### **Dataset**
#@markdown
#@markdown Defines `PushTStateDataset` and helper functions
#@markdown
#@markdown The dataset class
#@markdown - Load data (obs, action) from a zarr storage
#@markdown - Normalizes each dimension of obs and action to [-1,1]
#@markdown - Returns
#@markdown  - All possible segments with length `pred_horizon`
#@markdown  - Pads the beginning and the end of each episode with repetition
#@markdown  - key `obs`: shape (obs_horizon, obs_dim)
#@markdown  - key `action`: shape (pred_horizon, action_dim)

def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

# dataset
class PushTStateDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path,
                 pred_horizon, obs_horizon, action_horizon):

        # read from zarr dataset
        store = zarr.storage.ZipStore(ZARR_PATH, mode='r')
        dataset_root = zarr.open(store = store, mode = 'r')
        # All demonstration episodes are concatinated in the first dimension N
        train_data = {
            # (N, action_dim)
            'action': dataset_root['data']['action'][:],
            # (N, obs_dim)
            'obs': dataset_root['data']['state'][:]
        }
        # Marks one-past the last index for each episode
        episode_ends = dataset_root['meta']['episode_ends'][:]

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            # add padding such that each timestep in the dataset are seen
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        # all possible segments of the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        nsample['obs'] = nsample['obs'][:self.obs_horizon,:]
        return nsample


# ============================================================
# CELL 5: Dataset loading (identical to original)
# ============================================================
dataset_path = "pusht_cchi_v7_replay.zarr.zip"
if not os.path.isfile(dataset_path):
    id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
    gdown.download(id=id, output=dataset_path, quiet=False)

pred_horizon = 16
obs_horizon = 2
action_horizon = 8

dataset = PushTStateDataset(
    dataset_path=dataset_path,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)
stats = dataset.stats

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    num_workers=1,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True
)

batch = next(iter(dataloader))
print("batch['obs'].shape:", batch['obs'].shape)
print("batch['action'].shape", batch['action'].shape)


# ============================================================
# CELL 6: Heat Dissipation — 1D Forward Process
# ============================================================
class HeatDissipation1D:
    """
    1D heat equation applied to action trajectories.

    Given action trajectory a(t) of shape [B, T, D]:
      - Blur along the time axis T using FFT
      - High-frequency temporal structure (sharp accelerations, quick changes)
        dies first
      - Low-frequency structure (overall direction of motion) persists
      - At t=T_max, trajectory becomes a flat line (constant action)

    This is the 1D analog of our image heat equation,
    with time-axis blurring instead of spatial blurring.
    """

    def __init__(self, num_timesteps=100, alpha_max=8.0):
        self.T = num_timesteps
        # sqrt schedule so attenuation is uniform across timesteps
        self.sigmas = torch.sqrt(
            torch.linspace(0.01**2, alpha_max**2, num_timesteps)
        )
        self._filters = {}

    def _get_filter(self, sigma, seq_len, device):
        key = (round(sigma, 6), seq_len)
        if key not in self._filters:
            freq = torch.fft.fftfreq(seq_len, device=device)
            filt = torch.exp(-2 * (math.pi ** 2) * sigma**2 * freq**2)
            self._filters[key] = filt
        return self._filters[key]

    def blur(self, x, t_index):
        """
        Blur action trajectory along time axis.
        x: [B, T, D] — batch of action trajectories
        t_index: int — degradation level
        Returns: [B, T, D] — blurred trajectories
        """
        sigma = self.sigmas[t_index].item()
        B, T, D = x.shape
        filt = self._get_filter(sigma, T, x.device)  # [T]

        # FFT along time axis for each action dimension
        x_perm = x.permute(0, 2, 1)  # [B, D, T]
        x_freq = torch.fft.fft(x_perm, dim=-1)
        x_filtered = x_freq * filt[None, None, :]
        result = torch.fft.ifft(x_filtered, dim=-1).real
        return result.permute(0, 2, 1)  # [B, T, D]

    def blur_batch(self, x, t):
        """Blur each trajectory by its own timestep."""
        out = torch.zeros_like(x)
        for ti in t.unique():
            mask = t == ti
            out[mask] = self.blur(x[mask], ti.item())
        return out


# ============================================================
# CELL 7: Network — 1D UNet for trajectory restoration
# ============================================================
# We reuse the EXACT SAME ConditionalUnet1D architecture from the
# diffusion policy notebook. The only difference is what it predicts:
#   Diffusion: predicts noise ε added to actions
#   Heat:      predicts original clean action a_0 from blurred a_t
#
# The architecture handles both cases — it's a 1D U-Net with
# FiLM conditioning on observation + degradation step.

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim,
                 kernel_size=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups),
        ])
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim,
                 diffusion_step_embed_dim=256,
                 down_dims=[256, 512, 1024],
                 kernel_size=5, n_groups=8):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim,
                                       kernel_size=kernel_size, n_groups=n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim,
                                       kernel_size=kernel_size, n_groups=n_groups),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim,
                                           kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim=cond_dim,
                                           kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim=cond_dim,
                                           kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim=cond_dim,
                                           kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters())))

    def forward(self, sample, timestep, global_cond=None):
        sample = sample.moveaxis(-1, -2)
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long,
                                     device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = x.moveaxis(-1, -2)
        return x


# ============================================================
# CELL 8: Setup
# ============================================================
obs_dim = 5
action_dim = 2
num_heat_steps = 50  # blur levels (vs 100 diffusion steps in original)

# Create heat dissipation process
heat = HeatDissipation1D(num_timesteps=num_heat_steps, alpha_max=6.0)

# Create network — SAME architecture as diffusion policy
# But this one predicts clean actions (not noise)
action_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim * obs_horizon
)

device = torch.device('cuda')
_ = action_pred_net.to(device)


# ============================================================
# CELL 9: Verify heat dissipation on real trajectories
# ============================================================
print("\n" + "=" * 60)
print("HEAT DISSIPATION ON ACTION TRAJECTORIES")
print("=" * 60)

sample_actions = next(iter(dataloader))['action'][:4].to(device)
print(f"Sample trajectory shape: {sample_actions.shape}")  # [4, 16, 2]

# Check mass conservation
for t_idx in [0, num_heat_steps // 4, num_heat_steps // 2, num_heat_steps - 1]:
    blurred = heat.blur(sample_actions, t_idx)
    orig_mass = sample_actions.sum().item()
    blur_mass = blurred.sum().item()
    ratio = blur_mass / orig_mass if orig_mass != 0 else float('nan')
    print(f"  t={t_idx:3d}: mass ratio = {ratio:.6f}")

# Show what blurring does to a trajectory
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(16, 6))
t_levels = [0, num_heat_steps // 6, num_heat_steps // 3,
            num_heat_steps // 2, 2 * num_heat_steps // 3,
            5 * num_heat_steps // 6, num_heat_steps - 1]
# Show trajectory 0, both action dims
traj = sample_actions[0]  # [16, 2]
for i, t_idx in enumerate(t_levels[:4]):
    blurred = heat.blur(sample_actions[:1], t_idx)[0]  # [16, 2]
    axes[0, i].plot(traj[:, 0].cpu(), 'b-', alpha=0.3, label='original')
    axes[0, i].plot(blurred[:, 0].cpu(), 'r-', label='blurred')
    axes[0, i].set_title(f't={t_idx}')
    axes[0, i].legend(fontsize=7)
    axes[1, i].plot(traj[:, 1].cpu(), 'b-', alpha=0.3)
    axes[1, i].plot(blurred[:, 1].cpu(), 'r-')
axes[0, 0].set_ylabel('Action dim 0')
axes[1, 0].set_ylabel('Action dim 1')
fig.suptitle('Heat dissipation on action trajectories (coarse → flat)')
plt.tight_layout()
plt.savefig('heat_action_forward.png', dpi=150)
plt.close()
print("Saved heat_action_forward.png")


# ============================================================
# CELL 10: Training — Heat Dissipation Policy
# ============================================================

import copy

def eval_episode(net, seed=100000, max_steps=200):
    """Run one eval episode and return (imgs, max_reward)."""
    eval_env = PushTEnv()
    eval_env.seed(seed)
    obs, _ = eval_env.reset()
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    imgs = [eval_env.render(mode='rgb_array')]
    rewards = []
    done = False
    step_idx = 0

    net.eval()
    with torch.no_grad():
        while not done:
            obs_seq = np.stack(obs_deque)
            nobs = normalize_data(obs_seq, stats=stats['obs'])
            nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)
            obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

            naction = 0.01 * torch.randn(
                (1, pred_horizon, action_dim), device=device)

            for k in reversed(range(num_heat_steps)):
                t_batch = torch.full((1,), k, device=device, dtype=torch.long)
                action_pred = net(
                    sample=naction, timestep=t_batch, global_cond=obs_cond)
                if k > 0:
                    naction = heat.blur(action_pred.clamp(-1, 1), k - 1)
                else:
                    naction = action_pred

            naction = naction.cpu().numpy()[0]
            action_pred_unnorm = unnormalize_data(naction, stats=stats['action'])
            start = obs_horizon - 1
            action = action_pred_unnorm[start:start + action_horizon, :]

            for a in action:
                obs, reward, done, _, _ = eval_env.step(a)
                obs_deque.append(obs)
                rewards.append(reward)
                imgs.append(eval_env.render(mode='rgb_array'))
                step_idx += 1
                if step_idx > max_steps or done:
                    done = True
                    break
    net.train()
    return imgs, max(rewards) if rewards else 0.0


print("\n" + "=" * 60)
print("TRAINING: Heat Dissipation Policy")
print("=" * 60)

num_epochs = 100

ema = EMAModel(parameters=action_pred_net.parameters(), power=0.75)

optimizer = torch.optim.AdamW(
    params=action_pred_net.parameters(),
    lr=1e-4, weight_decay=1e-6)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=len(dataloader) * num_epochs)

with tqdm(range(num_epochs), desc='Epoch') as tglobal:
    for epoch_idx in tglobal:
        epoch_loss = list()
        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                nobs = nbatch['obs'].to(device)
                naction = nbatch['action'].to(device)
                B = nobs.shape[0]

                obs_cond = nobs[:, :obs_horizon, :].flatten(start_dim=1)

                timesteps = torch.randint(
                    0, num_heat_steps, (B,), device=device).long()

                blurred_actions = heat.blur_batch(naction, timesteps)

                action_pred = action_pred_net(
                    blurred_actions, timesteps, global_cond=obs_cond)

                loss = F.mse_loss(action_pred, naction)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                ema.step(action_pred_net.parameters())

                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)
        tglobal.set_postfix(loss=np.mean(epoch_loss))

        # --- per-epoch video using current EMA weights ---
        eval_net = copy.deepcopy(action_pred_net)
        ema.copy_to(eval_net.parameters())
        epoch_imgs, epoch_score = eval_episode(eval_net)
        video_path = f'vis_heat_epoch_{epoch_idx + 1:02d}.mp4'
        vwrite(video_path, epoch_imgs)
        print(f"  [epoch {epoch_idx + 1}] score={epoch_score:.3f}  saved {video_path}")
        del eval_net

# EMA weights for inference
ema_action_pred_net = action_pred_net
ema.copy_to(ema_action_pred_net.parameters())
print("Training complete!")


# ============================================================
# CELL 11: Inference — Heat Dissipation Policy (final eval)
# ============================================================
print("\n" + "=" * 60)
print("INFERENCE: Heat Dissipation Policy on PushT")
print("=" * 60)

imgs, score = eval_episode(ema_action_pred_net)
print('Score: ', score)
vwrite('vis_heat.mp4', imgs)
# Video('vis_heat.mp4', embed=True, width=256, height=256)


# ============================================================
# CELL 12: Analysis — What to compare
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS: Metrics to compare with Diffusion Policy")
print("=" * 60)

print("""
WHAT TO LOOK FOR:

1. SCORE (task success):
   - Diffusion Policy typically gets 0.7-0.9 on PushT
   - Compare your heat policy score above

2. TRAJECTORY SMOOTHNESS:
   Run this after both policies have generated trajectories:
""")

# Measure smoothness of generated actions
action_sequence = np.array([unnormalize_data(
    normalize_data(np.stack(list(obs_deque)), stats['obs']),
    stats['obs']) for _ in range(1)])

# Jerk = d³x/dt³ (third derivative of position)
# Lower jerk = smoother trajectory
all_actions = np.array(rewards)  # use rewards as proxy for analysis

print("""
3. INFERENCE SPEED:
   - Diffusion: 100 denoising steps per action chunk
   - Heat: {} blur-reversal steps per action chunk
   - Heat should be faster if fewer steps suffice

4. COARSE-TO-FINE INSPECTION:
   - Stop the reverse process at different points
   - Early stop = coarse plan (general direction)
   - Full reversal = fine-grained trajectory
   - This is FREE with heat policy, impossible with diffusion

5. TRAJECTORY CONSISTENCY:
   - Compare the variance of generated trajectories across episodes
   - Heat policy should show less jitter between re-planning steps
   - Look at the video: does the agent move more smoothly?

6. Try reducing num_heat_steps to 20 or 10 and re-run.
   Heat policy may maintain quality with far fewer steps
   because the coarse-to-fine hierarchy means early steps
   contribute more than later refinement steps.
""".format(num_heat_steps))