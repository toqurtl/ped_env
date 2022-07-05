from gym.envs.registration import register

register(
    id='pysfrl_sim-v0',
    entry_point='pysfrl_sim.envs:PysfrlEnv',
)