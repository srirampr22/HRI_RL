import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from HumanRobot_env import HumanRobotEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.utils import get_device
import os
import torch
import argparse
from typing import Callable

def parse_args():
    
    parser = argparse.ArgumentParser(description="Train PPO model with custom hyperparameters")
    parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps to run for each environment per update")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for each gradient update")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs to run for each update")
    parser.add_argument("--clip_range", type=float, default=0.2, help="Clipping parameter for PPO")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--train_steps", type=int, default=100000, help="Total number of training steps")
    parser.add_argument("--trans_wei", type=int, default=3.0, help="Tranlation reward weight")
    parser.add_argument("--rot_wei", type=int, default=2.0, help="rotation reward weight")
    parser.add_argument("--vel_wei", type=int, default=1.0, help="velocity reward weight")
    parser.add_argument("--col_wei", type=int, default=2.0, help="collision reward weight")
    parser.add_argument("--goal_wei", type=int, default=0.0, help="goal reward weight")
    parser.add_argument("--train_env", type=int, default=4, help="no of envs that can be run in parallel during training")
    parser.add_argument("--eval_env", type=int, default=1, help="no of envs that can be run in parallel during evaluvation")
    parser.add_argument("--reward_thresh", type=int, default=4000, help="no of envs that can be run in parallel during evaluvation")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use for training (cpu or cuda)")

    return parser.parse_args()

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def get_device(device):
    """Gets the device (GPU if any) and logs the type"""
    if torch.cuda.is_available() and device == 'cuda':
        device = torch.device("cuda")
        print(f"Found GPU device: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("No GPU found: Running on CPU")
    return device

def setup(device, args, env_kwargs, lr_scheduler=None):
    
    if lr_scheduler is None:
        lr_scheduler = args.learning_rate  
    

    # Create a vectorized environment with 4 parallel instances of HumanRobotEnv
    train_env = make_vec_env(lambda: HumanRobotEnv(**env_kwargs), n_envs=args.train_env, seed=0) # Was not passing reward_weights proeprely
    eval_env = make_vec_env(lambda: HumanRobotEnv(reward_weights=reward_weights), n_envs=args.eval_env, seed=0)
    
    # model = PPO(ActorCriticPolicy, env, learning_rate= 0.1,verbose=2, ent_coef=0.7,clip_range=0.2, device=device)
    # model = PPO("MlpPolicy", env, verbose=2, tensorboard_log="./ppo_human_robot_tensorboard/", device=device)
    
    model = PPO(
        "MlpPolicy", train_env, verbose=1, tensorboard_log="./ppo_human_robot_tensorboard/", device=device,
        learning_rate=lr_scheduler,  
        n_steps=args.n_steps,  
        batch_size=args.batch_size, 
        n_epochs=args.n_epochs, 
        clip_range=args.clip_range, 
        ent_coef=args.ent_coef
    )
    
    return model, train_env, eval_env

def main():
    args = parse_args()
    reward_weights = {'translation': args.trans_wei, 'rotation': args.rot_wei, 'velocity': args.vel_wei, 'collision': args.col_wei, 'goal': args.goal_wei}
    
    env_kwargs = {'reward_weights': reward_weights}

    device = get_device(device=args.device)
    
    lr_scheduler = linear_schedule(args.learning_rate)
    
    
    model, train_env, eval_env = setup(device, args, env_kwargs, lr_scheduler)
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=args.reward_thresh, verbose=1)
    
    # evaluation callback to periodically evaluate the model
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/best_model', 
                                 callback_on_new_best=callback_on_best,
                                 log_path='./logs/', eval_freq=max(500 // args.train_env, 1),n_eval_episodes=5,
                                 deterministic=True, render=False)
    
    # # Training
    model.learn(total_timesteps=args.train_steps, callback=[eval_callback], progress_bar=True)
    
    # Save the trained model
    model_save_path = "/home/sriram/gym_play/trained_models/ppo_human_robot"
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # # Check if the model was saved successfully
    # if os.path.exists(model_save_path + ".zip"):
    #     print("Model successfully saved.")
    # else:
    #     print("Model save failed.")

    # # Verify if the model is using CUDA
    # if next(model.policy.parameters()).is_cuda:
    #     print("Model is using CUDA")
    # else:
    #     print("Model is using CPU")

if __name__ == "__main__":
    main()
