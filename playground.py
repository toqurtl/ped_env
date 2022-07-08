
from pysfrl.experiment.exp_setting import ExpSetting
from pysfrl.data.video_data import VideoData

from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np


from stable_baselines3 import PPO
import os
import json

from pysfrl.rl.env import PysfrlEnv

onedrive_path = os.environ['onedrive']
exp_folder_path = os.path.join(onedrive_path, "연구\\pandemic\\experiment\\0707_nn")
video_folder_path = os.path.join(onedrive_path, "연구\\pandemic\\data\\ped_texas\\new\\new_opposite")

exp = ExpSetting(exp_folder_path=exp_folder_path)


env = PysfrlEnv(exp.get_simulator_list())
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")
model.learn(total_timesteps=1000000, tb_log_name="first_run")
model.save("model/test_model")
# # print('evaluate')
# model = PPO.load("model/test_model.zip", env=env)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")