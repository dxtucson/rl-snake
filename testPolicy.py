from tf_agents.environments import TFPyEnvironment
from tf_agents.policies import PolicySaver, policy_saver
import tensorflow as tf
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
import numpy as np

from SnakeEnv import SnakeEnv

policy_dir = 'Policy_2023-07-17_80000'
saved_policy = tf.saved_model.load(policy_dir)

env = SnakeEnv()
env.seed(43)
env.reset()
np.random.seed(42)
tf_env = TFPyEnvironment(env)

total_score2 = 0


def save_frames(trajectory):
    global total_score2
    total_score2 += tf_env.pyenv.current_time_step().reward[0]



watch_driver = DynamicStepDriver(
    tf_env,
    saved_policy,
    observers=[save_frames],
    num_steps=10000)
final_time_step, final_policy_state = watch_driver.run()
print(f'\nafter training score: {total_score2}')
