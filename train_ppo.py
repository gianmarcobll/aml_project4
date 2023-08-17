"""Sample script for training a control policy on the Hopper environment, using PPO algorithm.
We are searching for the optimal value for the learning rate
"""
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os


def main():
    env = gym.make('CustomHopper-source-v0')
    env = DummyVecEnv([lambda: env])
    
    target_env = gym.make('CustomHopper-target-v0')
    target_env = DummyVecEnv([lambda: target_env])

    # hyperparameter: learning rate
    learning_rates = [1e-3, 3e-4, 1e-4, 1e-5, 1e-6]
    training_episodes =  [100_000, 250_000, 500_000]
    i=0
        
    for lr in learning_rates:
        for te in training_episodes:
            print(f"--- Training PPO on SOURCE environment (Learning Rate = {lr})--- Training episodes = {te} ")      
            if os.path.exists(f"training/models/PPO/source_LR_{i}_TE_{te}.zip"):
                print("Found source model!")
                model = PPO.load(f"training/models/PPO/sorce_LR{i}", env=env)
            else:
                print("source model file not found. training...")
                model = PPO("MlpPolicy", env, verbose = 1, learning_rate = lr)
                model.learn(total_timesteps = te, progress_bar = True)
                model.save(f"training/models/PPO/source_LR_{i}_TE_{te}")
            print(f"--- Training PPO on TARGET environment (Learning Rate = {lr})--- Training episodes = {te} ")
            if os.path.exists(f"training/models/PPO/target_LR_{i}_TE_{te}.zip"):
                print("Found source model!")
                model_target = PPO.load(f"training/models/PPO/target_LR{i}", env=env)
            else:
                print("target model file not found. training...")
                model_target = PPO("MlpPolicy", target_env, verbose = 1, learning_rate = lr)
                model_target.learn(total_timesteps = te, progress_bar = True)
                model_target.save(f"training/models/PPO/source_LR_{i}_TE_{te}")


            print("Source-Source environment results:")
            print(evaluate_policy(model, env, n_eval_episodes=50, render=False))
            
            print("Source-Target environment results:")
            print(evaluate_policy(model, target_env, n_eval_episodes=50, render=False))

            print("Target-Target environment results:")
            print(evaluate_policy(model_target, target_env, n_eval_episodes=50, render=False))
            i+=1

            env.close()

if __name__ == '__main__':
    main()