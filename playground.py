
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
# entry_path = os.path.abspath(".")
# default_cfg_path = os.path.join(entry_path, "pysfrl", "test", "data", "simulation_nn_config.json")
# exp.set_default_cfg_path(default_cfg_path)

# for vid_id in os.listdir(video_folder_path):
#     vid_path = os.path.join(video_folder_path, vid_id)
#     video_data = VideoData(scene_folder=vid_path)
#     exp.add_scene_from_video(video_data, vid_id)

sim = exp.get_simulator("29")
# sim.simulate()
# exit()
env = PysfrlEnv([sim])
model = PPO.load("model/test_model.zip", env=env)
obs = env.reset()
while True:
    action, states = model.predict(obs)
    obs, reward, done, info = env.step(action)    
    if done:
        s = env.simulator
        # sim_result_path = os.path.join(".", "sim_result.json")
        # fig_path = os.path.join(".", "trajectory.png")
        # SimResult.sim_result_to_json(s, sim_result_path)
        # fig, ax = PlotGenerator.generate_sim_result_plot((-5,5,-10,10), s)    
        # fig.savefig(fig_path)
        # break
        break

