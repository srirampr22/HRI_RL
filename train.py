import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from HumanRobot_env import HumanRobotEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_device
import os
import torch

def parse_args():
    
    parser = argparse.ArgumentParser(description="Train PPO model with custom hyperparameters")
    parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps to run for each environment per update")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for each gradient update")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs to run for each update")
    parser.add_argument("--clip_range", type=float, default=0.2, help="Clipping parameter for PPO")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--train_steps", type=int, default=500000, help="Total number of training steps")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use for training (cpu or cuda)")

    return parser.parse_args()

def setup(args):
    
    # Create a vectorized environment with 4 parallel instances of HumanRobotEnv
    env = make_vec_env(lambda: HumanRobotEnv(), n_envs=4)

    # Check the environment to ensure it adheres to the Gymnasium API
    single_env = HumanRobotEnv()
    check_env(single_env)
    print("Environment is valid")
    
    # model = PPO(ActorCriticPolicy, env, learning_rate= 0.1,verbose=2, ent_coef=0.7,clip_range=0.2, device=device)
    # model = PPO("MlpPolicy", env, verbose=2, tensorboard_log="./ppo_human_robot_tensorboard/", device=device)
    
    model = PPO(
        "MlpPolicy", env, verbose=2, tensorboard_log="./ppo_human_robot_tensorboard/", device=device,
        learning_rate=args.learning_rate,  
        n_steps=args.n_steps,  
        batch_size=args.batch_size, 
        n_epochs=args.n_epochs, 
        clip_range=args.clip_range, 
        ent_coef=args.ent_coef
    )
    
    return model, env

def main():
    args = parse_args()
    # print(args)
    
    device = get_device(device=args.device)
    print(device)
    
    
    model, vec_env = setup(args)
    n_steps = 100
    train_steps = 500000
    
    # callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=2)
    
    # evaluation callback to periodically evaluate the model
    eval_callback = EvalCallback(vec_env, best_model_save_path='./logs/best_model', 
                                #  callback_on_new_best=callback_on_best,
                                 log_path='./logs/', eval_freq=500,
                                 deterministic=True, render=False)
    
    # Training
    model.learn(total_timesteps=train_steps, callback=[eval_callback], progress_bar=True)
    
    # Save the trained model
    model_save_path = "/home/sriram/gym_play/temp"
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Check if the model was saved successfully
    if os.path.exists(model_save_path + ".zip"):
        print("Model successfully saved.")
    else:
        print("Model save failed.")

    # # Verify if the model is using CUDA
    # if next(model.policy.parameters()).is_cuda:
    #     print("Model is using CUDA")
    # else:
    #     print("Model is using CPU")

if __name__ == "__main__":
    main()
