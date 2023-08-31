import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

def main():
    env = gym.make('CustomHopper-source-v0')
    
    target_env = gym.make('CustomHopper-target-v0')

    # hyperparameter: learning rate
    learning_rates = [1e-3, 3e-4, 1e-4, 1e-5, 1e-6]
    i=0
    eval_freq = 10_000
    for lr in learning_rates:
        save_path_source_source = f"./training/logs/LR_{i}_source_source"
        save_path_source_target = f"./training/logs/LR_{i}_source_target"
        save_path_target_target = f"./training/logs/LR_{i}_target_target"
        print("source source LR:", lr)
        eval_callback = EvalCallback(eval_env=env, eval_freq=eval_freq, log_path=save_path_source_source, n_eval_episodes=50)
        model = PPO("MlpPolicy", env, verbose = 1, learning_rate = lr)
        model.learn(total_timesteps = 500_000, callback=eval_callback, progress_bar = True)

        print("source target LR:", lr)
        eval_callback = EvalCallback(eval_env=target_env, eval_freq=eval_freq, log_path=save_path_source_target, n_eval_episodes=50)
        model = PPO("MlpPolicy", env, verbose = 1, learning_rate = lr)
        model.learn(total_timesteps = 500_000, callback=eval_callback, progress_bar = True)

        print("target target LR:", lr)
        eval_callback = EvalCallback(eval_env=target_env, eval_freq=eval_freq, log_path=save_path_target_target, n_eval_episodes=50)
        model = PPO("MlpPolicy", target_env, verbose = 1, learning_rate = lr)
        model.learn(total_timesteps = 500_000, callback=eval_callback, progress_bar = True)
        i+=1
        
    env.close()
    target_env.close()

if __name__ == '__main__':
    main()