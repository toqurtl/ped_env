
from pysfrl.data.video_data import VideoData
from pysfrl.experiment.exp_setting import ExpSetting
from pysfrl.rl.env import PysfrlEnv
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3 import PPO
import os
import torch

# GPU 할당 변경하기
GPU_NUM = 0 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU

print ('# Current cuda device: ', torch.cuda.current_device()) # check

entry_path = os.path.abspath(".")
RL_CFG_PATH = os.path.join(entry_path, "pysfrl", "test", "data", "simulation_nn_config.json")

onedrive_path = os.environ['onedrive']
exp_folder_path = os.path.join(onedrive_path, "연구\\pandemic\\experiment\\0708")
exp_folder_path_2 = os.path.join(onedrive_path, "연구\\pandemic\\experiment\\0708_2")
video_folder_path = os.path.join(onedrive_path, "연구\\pandemic\\data\\ped_texas\\new\\whole")


exp_2 = ExpSetting(exp_folder_path=exp_folder_path_2)


# exp = ExpSetting(exp_folder_path=exp_folder_path)

# exp.set_default_cfg_path(RL_CFG_PATH)

# exp.simulate_scene("29")
# exp_2.simulate_scene("29")
env = PysfrlEnv(exp_2.get_simulator_list())
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")
model.learn(total_timesteps=150000, tb_log_name="0708_2")
model.save("model/0708_3")
# # # print('evaluate')
# # model = PPO.load("model/test_model.zip", env=env)

# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
# print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")