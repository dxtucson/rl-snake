import os

import numpy as np
from tf_agents.agents import DqnAgent
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.policies import PolicySaver, policy_saver
from SnakeEnv import SnakeEnv
import tensorflow as tf
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer

env = SnakeEnv()
env.seed(43)
env.reset()
np.random.seed(42)
tf_env = TFPyEnvironment(env)
fc_layer_params = [300, 80, 20]

q_net = QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    fc_layer_params=fc_layer_params)
train_step = tf.Variable(0)
update_period = 10  # run a training step every 4 collect steps
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

dataset = replay_buffer.as_dataset(
    sample_batch_size=20,
    num_steps=2,
    num_parallel_calls=3).prefetch(3)

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


train_agent(n_iterations=2000)
total_score2 = 0


def save_frames(trajectory):
    global total_score2
    total_score2 += tf_env.pyenv.current_time_step().reward[0]


policy_dir = 'policy_1999'
trained_policy = policy_saver.PolicySaver(agent.policy)
saved_policy = tf.saved_model.load(policy_dir)

input("Press Enter to continue...")
watch_driver = DynamicStepDriver(
    tf_env,
    trained_policy,
    observers=[save_frames],
    num_steps=200)
final_time_step, final_policy_state = watch_driver.run()
print(f'\nafter training score: {total_score2}')
