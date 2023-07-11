# This is a sample Python script.
from random import randint

import numpy as np
import tf_agents
from tf_agents.agents import DqnAgent
from tf_agents.environments import wrappers
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.policies import PolicySaver

from SnakeEnv import SnakeEnv
from tf_agents.environments import utils

import tensorflow as tf
from tensorflow import keras

from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer

env = SnakeEnv()
env.seed(43)
env.reset()
np.random.seed(42)
#env.step(1)
#print(env.current_time_step())
tf_env = TFPyEnvironment(env)
# preprocessing_layer = keras.layers.Lambda(
#     lambda obs: tf.cast(obs, np.float32) / 255.)
# conv_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
# import time
# total_score1 = 0
# for step in range(1000):
#     time_step = env.step(randint(0, 3))
#     time.sleep(0.2)
#     print(time_step)
#     total_score1 += time_step.reward
#
# print(total_score1)
#
# while True:
#     time.sleep(10)
#conv_layer_params=[(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
fc_layer_params=[200,50]

q_net = QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    #conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params)
train_step = tf.Variable(0)
update_period = 4  # run a training step every 4 collect steps
agent = DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network=q_net,
    gamma=0.95,
    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3),
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step)

agent.initialize()
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=100000)  # reduce if OOM error

replay_buffer_observer = replay_buffer.add_batch


class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total

    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")


from tf_agents.metrics import tf_metrics

train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric()
]

print('Qagent start to train')
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver

collect_driver = DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=update_period)  # collect 4 steps for each training iteration

from tf_agents.policies.random_tf_policy import RandomTFPolicy

# initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(),
#                                         tf_env.action_spec())
# init_driver = DynamicStepDriver(
#     tf_env,
#     initial_collect_policy,
#     observers=[replay_buffer.add_batch, ShowProgress(200)],
#     num_steps=200)  # <=> 80,000 ALE frames
# final_time_step, final_policy_state = init_driver.run()
#
# tf.random.set_seed(9)  # chosen to show an example of trajectory at the end of an episode

# trajectories, buffer_info = replay_buffer.get_next( # get_next() is deprecated
#    sample_batch_size=2, num_steps=3)

# trajectories, buffer_info = next(iter(replay_buffer.as_dataset(
#     sample_batch_size=2,
#     num_steps=3,
#     single_deterministic_pass=False)))
#
# print(trajectories.observation.shape)

dataset = replay_buffer.as_dataset(
    sample_batch_size=20,
    num_steps=2,
    num_parallel_calls=3).prefetch(3)

from tf_agents.utils.common import function

# collect_driver.run = function(collect_driver.run)
# agent.train = function(agent.train)

from tf_agents.eval.metric_utils import log_metrics
import logging

logging.getLogger().setLevel(logging.INFO)
log_metrics(train_metrics)



def train_agent(n_iterations):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print("\r{} loss:{:.5f}".format(
            iteration, train_loss.loss.numpy()), end="")
        if iteration % 1000 == 0:
            log_metrics(train_metrics)
            my_policy = agent.collect_policy

    saver = PolicySaver(my_policy, batch_size=None)
    saver.save('policy_%d' % iteration)

train_agent(n_iterations=10000)

total_score2 = 0


def save_frames(trajectory):
    global total_score2
    total_score2 += tf_env.pyenv.current_time_step().reward[0]


input("Press Enter to continue...")
watch_driver = DynamicStepDriver(
    tf_env,
    agent.policy,
    observers=[save_frames],
    num_steps=10000)
final_time_step, final_policy_state = watch_driver.run()
print(f'\nafter training score: {total_score2}')
