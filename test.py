import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from HumanRobot_env import HumanRobotEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_device
import os
import torch



def setup(device, model_path):
    # Create a vectorized environment with 4 parallel instances of HumanRobotEnv
    env = make_vec_env(lambda: HumanRobotEnv(), n_envs=1)
    # env = HumanRobotEnv()

    # # Check the environment to ensure it adheres to the Gymnasium API
    # single_env = HumanRobotEnv()
    # check_env(single_env)
    # print("Environment is valid")
    

    model = PPO.load(model_path, env=env, device=device)
    
    return model, env

def main():
    # args = parse_args()
    # print(args)
    model_path = "/home/sriram/gym_play/trained_models/ppo_human_robot.zip"
    
    device = get_device(device='cuda')
    print(device)
    
    model, env = setup(device, model_path)
    n_steps = 400
    
    # Evaluate the loaded model
    obs = env.reset()
    for _ in range(n_steps):  
        action, _states = model.predict(obs)
        obs, rewards, dones, infos = env.step(action) # weird techincal bug story thing for interviews, apprarently when you make_vec_env your env, it dosent care about the what your step returns (terminated, truncated), since stablebaseline3 is not updated to gymnassium API, it returns (done) instead like how it was in the old gym API, which is BS bro.
        # state, reward, terminated, truncated, info = env.step(action)
        # print("Reward:", rewards, "Info:", infos)
        env.envs[0].render()  
        # if terminated or truncated:
        #     normalized_state, reset_info = env.reset()
        #     print("Episode terminated, environment reset.")

    env.close()

if __name__ == "__main__":
    main()
