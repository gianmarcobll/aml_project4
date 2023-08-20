import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os

def main():
    env = gym.make('CustomHopper-source-v0')
    
    target_env = gym.make('CustomHopper-target-v0')

    # hyperparameter: learning rate
    learning_rates = [1e-3, 3e-4, 1e-4, 1e-5, 1e-6]
    # hyperparameter: training episodes
    training_episodes =  [100_000, 250_000, 500_000]
    i=0
    udr_eps = [0.3, 0.5, 0.7]
    masses_to_set = env.get_parameters()[1:]
    
    f = open("./results/training_ppo_udr_results.txt", "a")
    for lr in learning_rates:
        for te in training_episodes:
            for ue in udr_eps:
                lower_bound = masses_to_set * (1 - 0.5 * ue)
                upper_bound = masses_to_set * (1 + 0.5 * ue)
                distribution_set = [lower_bound[0], upper_bound[0], lower_bound[1], upper_bound[1], lower_bound[2], upper_bound[2]]
                env.set_udr_training(True)
                env.set_udr_distribution(distribution_set)
                print(f"--- Training PPO on SOURCE environment using UDR (Learning Rate = {lr}--- Training episodes = {te} --- UDR EPS = {ue}) ")      
                if os.path.exists(f"training/models/PPO_UDR/source_LR_{i}_TE_{te}_UDR_{str(int(ue*10))}.zip"):
                    print("Found source model!")
                    model = PPO.load(f"training/models/PPO_UDR/source_LR_{i}_TE_{te}_UDR_{str(int(ue*10))}", env=env)
                else:
                    print("model file not found. training...")
                    model = PPO("MlpPolicy", env, verbose = 1, learning_rate = lr)
                    model.learn(total_timesteps = te, progress_bar = True)
                    model.save(f"training/models/PPO_UDR/source_LR_{i}_TE_{te}_UDR_{str(int(ue*10))}")

                f.write(f"LR: {lr} TE: {te} UDR: {distribution_set}\n")
                print("Source-Source environment results:")
                res = evaluate_policy(model, env, n_eval_episodes=50, render=False)
                print(res)
                f.write("Source-Source environment results:")
                f.write(str(res)+"\n")
                
                print("Source-Target environment results:")
                res = evaluate_policy(model, target_env, n_eval_episodes=50, render=False)
                print(res)
                f.write("Source-Target environment results:")
                f.write(str(res)+"\n")
        i+=1
    env.close()
    target_env.close()
    f.close()

if __name__ == '__main__':
    main()