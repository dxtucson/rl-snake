from random import randint

import numpy as np
import tf_agents
from matplotlib import pyplot as plt
from tf_agents.agents import DqnAgent
from tf_agents.environments import wrappers
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork

from SnakeEnv import SnakeEnv
from tf_agents.environments import utils

import tensorflow as tf
from tensorflow import keras

from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from tf_agents.metrics import tf_metrics
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy

from tf_agents.utils.common import function
from tf_agents.eval.metric_utils import log_metrics
import logging

possible_actions = [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3],[0, 1, 2, 3]]

Q_values = np.full((8, 4), 0.0)  # -np.inf for impossible actions
# if action == 2, self._state[randint(0, 2)] = 1
transition_probabilities = [
    [[1,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0], [0,0.334,0.333,0,0.333,0,0,0], [1,0,0,0,0,0,0,0]], # 000-> 001, 010, 100
    [[0,1,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0.334,0,0.333,0,0.333,0,0], [0,1,0,0,0,0,0,0]],# 001-> 011, 101
    [[0,0,1,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0.334,0.333,0,0,0.333,0], [0,0,1,0,0,0,0,0]],# 010-> 011, 110
    [[0,0,0,1,0,0,0,0], [0,0,0,1,0,0,0,0], [0,0,0,0.667,0,0,0,0.333], [0,0,0,1,0,0,0,0]],# 011-> 111
    [[0,0,0,0,1,0,0,0], [0,0,0,0,1,0,0,0], [0,0,0,0,0.334,0.333,0.333,0], [0,0,0,0,1,0,0,0]],# 100-> 101,110
    [[0,0,0,0,0,1,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0.667,0,0.333], [0,0,0,0,0,1,0,0]],# 101-> 111
    [[0,0,0,0,0,0,1,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0.667,0.333], [0,0,0,0,0,0,1,0]],# 110-> 111,
    [[0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,1]]
]

rewards = [
    [[0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0]],
    [[0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0]],
    [[0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0]],
    [[0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0]],
    [[0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0]],
    [[0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0]],
    [[0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0]],
    [[0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0]],

]

gamma = 0.90  # the discount factor

history1 = [] # Not shown in the book (for the figure below)
for iteration in range(5000):
    Q_prev = Q_values.copy()
    history1.append(Q_prev) # Not shown
    for s in range(8):
        for a in possible_actions[s]:
            Q_values[s, a] = np.sum([
                    transition_probabilities[s][a][sp]
                    * (rewards[s][a][sp] + gamma * np.max(Q_prev[sp])) for sp in range(8)])

history1 = np.array(history1) # Not shown
print(Q_values)
print(np.argmax(Q_values, axis=1))


def step(state, action):
    probas = transition_probabilities[state][action]
    next_state = np.random.choice([0, 1, 2, 3,4,5,6,7], p=probas)
    reward = rewards[state][action][next_state]
    return next_state, reward

def exploration_policy(state):
    return np.random.choice(possible_actions[state])
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# To get smooth animations
import matplotlib.animation as animation
mpl.rc('animation', html='jshtml')

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rl"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

np.random.seed(42)

Q_values = np.full((8, 4), 0.0)


alpha0 = 0.05 # initial learning rate
decay = 0.005 # learning rate decay
gamma = 0.90 # discount factor
state = 0 # initial state
history2 = [] # Not shown in the book

for iteration in range(10000):
    history2.append(Q_values.copy()) # Not shown
    action = exploration_policy(state)
    next_state, reward = step(state, action)
    next_value = np.max(Q_values[next_state]) # greedy policy at the next step
    alpha = alpha0 / (1 + iteration * decay)
    Q_values[state, action] *= 1 - alpha
    Q_values[state, action] += alpha * (reward + gamma * next_value)
    state = next_state

history2 = np.array(history2) # Not shown

print(Q_values)
print(np.argmax(Q_values, axis=1))
true_Q_value = history1[-1, 0, 0]

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
axes[0].set_ylabel("Q-Value$(s_0, a_0)$", fontsize=14)
axes[0].set_title("Q-Value Iteration", fontsize=14)
axes[1].set_title("Q-Learning", fontsize=14)
for ax, width, history in zip(axes, (50, 10000), (history1, history2)):
    ax.plot([0, width], [true_Q_value, true_Q_value], "k--")
    ax.plot(np.arange(width), history[:, 0, 0], "b-", linewidth=2)
    ax.set_xlabel("Iterations", fontsize=14)
    ax.axis([0, width, 0, 1])

save_fig("q_value_plot")

