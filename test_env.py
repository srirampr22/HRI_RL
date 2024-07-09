import gym
from HumanRobot_env import HumanRobotEnv
from stable_baselines3.common.env_checker import check_env

def main():
    # Create the environment
    env = HumanRobotEnv()

    # It will check your custom environment and output additional warnings if needed
    # if check_env(env) is None:
    #     print("Environment is valid")
        
    # Reset the environment
    state, info = env.reset()

    # Visualize the environment
    for _ in range(1000):
        action = env.action_space.sample()  # Take random action
        state, reward, terminated, truncated, info = env.step(action)
        # print("Reward:", reward, "Info:", info)
        # print("State:", state, "Reward:", reward, "Info:", info)
        print("Terminated: ", terminated, "Truncated: ", truncated)
        env.render()
        if terminated or truncated:
            break
        
    env.close()


        
        
if __name__ == "__main__":
    main()