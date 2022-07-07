from pysfrl.test import test_simulator
from pysfrl.experiment.exp_setting import ExpSetting
from pysfrl.data.video_data import VideoData
from pysfrl.config.sim_config import SimulationConfig
from pysfrl.sim.simulator import Simulator
from pysfrl.sim.utils.sim_result import SimResult
from pysfrl.visualize.plots import PlotGenerator
from pysfrl.experiment import utils
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

from pysfrl.rl.env import PysfrlEnv
from stable_baselines3 import PPO
import os
import json

onedrive_path = os.environ['onedrive']
exp_folder_path = os.path.join(onedrive_path, "연구\\pandemic\\experiment\\0704")
video_folder_path = os.path.join(onedrive_path, "연구\\pandemic\\data\\ped_texas\\new\\new_opposite")



exp = ExpSetting(exp_folder_path=exp_folder_path)

# for scene_folder_path in exp.scene_folder_path_list():
#     a = utils.gt_trajectory_to_numpy(scene_folder_path)
#     xy_range = (-5, 5, -10, 10)
#     sub_plots = PlotGenerator.generate_sub_plots(xy_range)
#     sub_plots = PlotGenerator.plot_trajectory(sub_plots, a)
#     fig, ax = sub_plots
#     fig_path = os.path.join(scene_folder_path, "data", "gt_trajectory.png")
#     fig.savefig(fig_path)

# for vid_id in os.listdir(video_folder_path):
#     print(vid_id)
#     vid_path = os.path.join(video_folder_path, vid_id)
#     video_data = VideoData(scene_folder=vid_path)
#     exp.add_scene_from_video(video_data, vid_id)    


# exp.simulate_scene("29")
# simulator = exp.get_simulator("118")
# simulator.simulate()
# exit()
s = exp.get_simulator("64")

env = PysfrlEnv(exp.get_simulator_list())
# env = PysfrlEnv([s], 0)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")
model.learn(total_timesteps=10000, tb_log_name="first_run")
model.save("model/test_model")
# # print('evaluate')
# model = PPO.load("model/test_model.zip", env=env)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

# # model = PPO.load("model/test_model.zip", env=env)

# obs = env.reset()
# print(env.simulator.cfg.config_id)
# while True:
#     action, states = model.predict(obs)
#     obs, reward, done, info = env.step(action)    
#     if done:
#         s = env.simulator
#         sim_result_path = os.path.join(".", "sim_result.json")
#         fig_path = os.path.join(".", "trajectory.png")
#         SimResult.sim_result_to_json(s, sim_result_path)
#         fig, ax = PlotGenerator.generate_sim_result_plot((-5,5,-10,10), s)    
#         fig.savefig(fig_path)
#         break



