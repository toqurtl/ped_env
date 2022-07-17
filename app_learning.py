from pysfrl.experiment.exp_setting import ExpSetting
from pysfrl.data.video_data import VideoData
from pysfrl.sim.utils.sim_result import SimResult
from pysfrl.visualize.plots import PlotGenerator
from pysfrl.rl.env import PysfrlEnv
import numpy as np
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3 import PPO
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2

import os

onedrive_path = os.environ['onedrive']
exp_folder_path = os.path.join(onedrive_path, "연구\\pandemic\\experiment\\0708_2")
video_folder_path = os.path.join(onedrive_path, "연구\\pandemic\\data\\ped_texas\\new\\whole")

exp = ExpSetting(exp_folder_path=exp_folder_path)
sim_list = exp.get_simulator_list()

for sim in sim_list:
    cfg_idx = sim.cfg.config_id
    video_path = os.path.join(video_folder_path, cfg_idx)
    video_data = VideoData(scene_folder=video_path)
    sim.set_time_table(video_data.time_table)

env = PysfrlEnv(sim_list)
model = PPO2("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/0713")
model.learn(total_timesteps=100000, tb_log_name="0713")
model.save("model/0713_1")
