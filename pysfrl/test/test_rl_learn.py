from pysfrl.experiment.exp_setting import ExpSetting
from pysfrl.sim.utils.sim_result import SimResult
from pysfrl.visualize.plots import PlotGenerator
from pysfrl.rl.env import PysfrlEnv

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

import os


onedrive_path = os.environ['onedrive']
exp_folder_path = os.path.join(onedrive_path, "연구\\pandemic\\experiment\\0704")
video_folder_path = os.path.join(onedrive_path, "연구\\pandemic\\data\\ped_texas\\new\\new_opposite")


def test_learning():
    exp = ExpSetting(exp_folder_path=exp_folder_path)
    env = PysfrlEnv(exp.get_simulator_list())
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")
    model.learn(total_timesteps=10000, tb_log_name="first_run")
    model.save("model/test_model")
    # # print('evaluate')
    # model = PPO.load("model/test_model.zip", env=env)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")


def test_simulate_with_model():
    exp = ExpSetting(exp_folder_path=exp_folder_path)
    env = PysfrlEnv(exp.get_simulator_list())
    model = PPO.load("model/test_model.zip", env=env)

    obs = env.reset()
    print(env.simulator.cfg.config_id)
    while True:
        action, states = model.predict(obs)
        obs, reward, done, info = env.step(action)    
        if done:
            s = env.simulator
            sim_result_path = os.path.join(".", "sim_result.json")
            summary_path = os.path.join(".", "summary.json")
            fig_path = os.path.join(".", "trajectory.png")
            SimResult.sim_result_to_json(s, sim_result_path)
            SimResult.summary_to_json(s, summary_path)

            fig, ax = PlotGenerator.generate_sim_result_plot((-5,5,-10,10), s)    
            fig.savefig(fig_path)
            break



