import tkinter
from enum import Enum
from random import randint

import tensorflow as tf
import numpy as np
from tf_agents.environments import py_environment, wrappers
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import utils

WIDTH = 440
HEIGHT = 440
SNAKE_W = 20


class Status(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    DIED = 100


class SnakeEnv(py_environment.PyEnvironment):
    episode = 0
    snake_row = []
    snake_col = []
    snake_next_head_r = 0
    snake_next_head_c = 0
    reward = 0
    highest_score = 0
    reward_X = -1
    reward_Y = -1
    # will_hit_wall = 0
    # will_hit_self = 0
    root = tkinter.Tk()
    episode_str = tkinter.StringVar()
    episode_str.set('Generation: 0')
    score_str = tkinter.StringVar()
    score_str.set('Score: 0')
    status = Status.DIED
    canvas = tkinter.Canvas(root, bg='black', height=HEIGHT + 40, width=WIDTH)
    got_reward = False

    def __init__(self, seed=42):
        tf.random.set_seed(seed)
        self.root.geometry(f'{WIDTH}x{HEIGHT + 40}')
        # snake area
        for row in range(20):
            for col in range(20):
                rect_x = SNAKE_W + col * SNAKE_W
                rect_y = SNAKE_W + row * SNAKE_W
                self.canvas.create_rectangle(rect_x, rect_y, rect_x + SNAKE_W, rect_y + SNAKE_W,
                                             fill='black', outline='', tags='{},{}'.format(row, col))
        # border
        self.canvas.create_rectangle(0, 0, WIDTH, SNAKE_W, fill='red', outline='')
        self.canvas.create_rectangle(WIDTH - SNAKE_W, 0, WIDTH, HEIGHT, fill='red', outline='')
        self.canvas.create_rectangle(0, HEIGHT - SNAKE_W, WIDTH, HEIGHT, fill='red', outline='')
        self.canvas.create_rectangle(0, 0, SNAKE_W, HEIGHT, fill='red', outline='')
        episode_label = tkinter.Label(self.root, font='Arial 12 bold', fg='white', bg='black',
                                      textvariable=self.episode_str)
        episode_label.place(x=SNAKE_W, y=HEIGHT + 20, anchor='w')
        score_label = tkinter.Label(self.root, font='Arial 12 bold', fg='white', bg='black',
                                    textvariable=self.score_str)
        score_label.place(x=WIDTH - SNAKE_W, y=HEIGHT + 20, anchor='e')
        # add to window and show
        self.canvas.pack()
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        # 40 snake_x and 20 snake_y, 2 for snake head, 2 for reward xy, 1 for direction, 4 for death move
        min = [0] * 40 + [0, 0] + [0, 0] + [0] + [0, 0, 0, 0]
        max = [1] * 40 + [19, 19] + [19, 19] + [3] + [1, 1, 1, 1]
        self._observation_spec = array_spec.BoundedArraySpec(shape=(49,), dtype=np.int32, minimum=min,
                                                             maximum=max,
                                                             name='observation')
        self._state = [0] * 49
        self._episode_ended = False

    def update_state(self):
        self._state = [0] * 49
        for row in self.snake_row:
            self._state[row] = 1
        for col in self.snake_col:
            self._state[20 + col] = 1
        self._state[40] = self.snake_row[0]
        self._state[41] = self.snake_col[0]
        self._state[42] = self.reward_X
        self._state[43] = self.reward_Y
        self._state[44] = self.status.value
        if self.snake_row[0] == 0:
            self._state[45] = 1
            if self.status == Status.UP:
                self._state[47] = 1
        if self.snake_col[0] == 19:
            self._state[46] = 1
            if self.status == Status.RIGHT:
                self._state[48] = 1
        if self.snake_row[0] == 19:
            self._state[47] = 1
            if self.status == Status.DOWN:
                self._state[45] = 1
        if self.snake_col[0] == 1:
            self._state[48] = 1
            if self.status == Status.LEFT:
                self._state[46] = 1

        # self._state[46] = self.will_hit_wall
        # self._state[47] = self.will_hit_self
        # self._state[46] = self.status.value

    def head_will_hit_self(self):
        for i, r in enumerate(self.snake_row[0:-1]):
            if r == self.snake_next_head_r and self.snake_col[i] == self.snake_next_head_c:
                return True
        return False

    def head_hit_self(self):
        for i, r in enumerate(self.snake_row[1:]):
            if r == self.snake_row[0] and self.snake_col[1 + i] == self.snake_col[0]:
                return True
        return False

    def update_snake(self):
        head_row = self.snake_row[0]
        head_col = self.snake_col[0]
        if self.status == Status.UP:
            head_row -= 1
            self.snake_next_head_r = head_row - 1
            self.snake_next_head_c = head_col
        elif self.status == Status.RIGHT:
            head_col += 1
            self.snake_next_head_r = head_row
            self.snake_next_head_c = head_col + 1
        elif self.status == Status.DOWN:
            head_row += 1
            self.snake_next_head_r = head_row + 1
            self.snake_next_head_c = head_col
        elif self.status == Status.LEFT:
            head_col -= 1
            self.snake_next_head_r = head_row
            self.snake_next_head_c = head_col - 1
        # self.will_hit_wall = self.snake_next_head_r > 19 or self.snake_next_head_r < 0 or \
        #                      self.snake_next_head_c > 19 or self.snake_next_head_c < 0
        # self.will_hit_self = self.head_will_hit_self()
        # if 0 <= head_row < 20 and 0 <= head_col < 20 and not self.head_hit_self():
        if 0 <= head_row < 20 and 0 <= head_col < 20:
            self.snake_row.insert(0, head_row)
            self.snake_col.insert(0, head_col)
        else:
            self.status = Status.DIED

    def check_reward(self):
        if self.reward_X == self.snake_row[0] and self.reward_Y == self.snake_col[0]:
            self.reward += 1
            self.highest_score = max(self.reward, self.highest_score)
            self.episode_str.set(f'Generation: {self.episode}  Best Score: {self.highest_score}')
            self.got_reward = True
            self.score_str.set(f'Score: {self.reward}')
            self.create_reward()
        else:
            pop_r = self.snake_row.pop()
            pop_c = self.snake_col.pop()
            pop = self.canvas.find_withtag('{},{}'.format(pop_r, pop_c))
            self.canvas.itemconfig(pop[0], fill='black', outline='')

    def draw_snake(self):
        for i, r in enumerate(self.snake_row):
            all_items = self.canvas.find_withtag('{},{}'.format(self.snake_row[i], self.snake_col[i]))
            self.canvas.itemconfig(all_items[0], fill='white', outline='green', width='2')
        reward = self.canvas.find_withtag('{},{}'.format(self.reward_X, self.reward_Y))
        if reward:
            self.canvas.itemconfig(reward[0], fill='yellow')

    def create_reward(self):
        if len(self.snake_row) < 12 and self.episode < 800:
            reward_r = randint(7, 11)
            reward_c = randint(7, 11)
        else:
            reward_r = randint(0, 19)
            reward_c = randint(0, 19)
        for i, r in enumerate(self.snake_row):
            if self.snake_row[i] == reward_r and self.snake_col[i] == reward_c:
                reward_r = randint(0, 19)
                reward_c = randint(0, 19)
                break
        self.reward_X = reward_r
        self.reward_Y = reward_c

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.status = Status.RIGHT
        self.snake_row = [9, 9, 9, 9]
        self.snake_col = [4, 3, 2, 1]
        self.score_str.set('Score: 0')
        self.reward = 0
        self.reward_X = -1
        self.reward_Y = -1
        # self.will_hit_wall = 0
        # self.will_hit_self = 0
        self.snake_next_head_r = 9
        self.snake_next_head_c = 5
        self.episode += 1
        self.episode_str.set(f'Generation: {self.episode}  Best Score: {self.highest_score}')
        self._episode_ended = False
        self.got_reward = False
        for r in range(20):
            for c in range(20):
                items = self.canvas.find_withtag('{},{}'.format(r, c))
                self.canvas.itemconfig(items[0], fill='black', outline='')
        self.draw_snake()
        self.root.update()
        self.create_reward()
        self.root.update()
        self.update_state()
        return ts.restart(np.array(self._state, dtype=np.int32))

    def _step(self, action):
        if self.status != Status.DIED:
            if action == 0 and self.status != Status.DOWN:  # Up,
                self.status = Status.UP
            elif action == 1 and self.status != Status.LEFT:  # right
                self.status = Status.RIGHT
            elif action == 2 and self.status != Status.UP:  # down
                self.status = Status.DOWN
            elif action == 3 and self.status != Status.RIGHT:  # left
                self.status = Status.LEFT
            self.update_snake()
            if self.status == Status.DIED:
                self._episode_ended = True
                _ret = ts.termination(np.array(self._state, dtype=np.int32), -100)
                self.reset()
                return _ret
            self.check_reward()
            self.draw_snake()
            self.root.update()
            self.update_state()
            if self.got_reward:
                self.got_reward = False
                return ts.transition(np.array(self._state, dtype=np.int32),
                                     reward=100)
            elif self.head_hit_self():
                return ts.transition(np.array(self._state, dtype=np.int32),
                                     reward=-0.1)
            else:
                return ts.transition(np.array(self._state, dtype=np.int32),
                                     reward=0)

    def seed(self, seed):
        tf.random.set_seed(seed)
