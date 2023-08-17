from stable_baselines3.common.evaluation import evaluate_policy as evaluator
from policy_gradient import get_model
import gym
from wrappers import *

N_EVAL_EPISODES = 100
from env.custom_hopper import *

def evaluate_policy(params):
    evaluate_domain = params['evaluate_domain']
    env = gym.make(f'CustomHopper-{evaluate_domain}-v0')
    if params['vision_based']:
        env = gym.make(f"CustomHopper-{evaluate_domain}-v0")
        print(env.observation_space)
        env = PixelObservationWrapper(env)
        print('state observation shape', env.observation_space.shape)
        env = PreprocessWrapper(env)
        print('obervation shape after preprocessing:', env.observation_space.shape)
                
        env = GrayScaleWrapper(env, smooth=False, preprocessed=True, keep_dim=False)
        print('obervation shape after gray scaling:', env.observation_space.shape)
        env = ResizeWrapper(env, shape=[240, 240])
        print('obervation shape after resizing:', env.observation_space.shape)

        env = FrameStack(env, num_stack=4)
        print('obervation shape after stacking:', env.observation_space.shape)
    model = get_model(params['model_path'])
    mean_reward, std_reward = evaluator(env=env, model=model, n_eval_episodes=N_EVAL_EPISODES, render=False)
    print("mean reward: ", mean_reward, "std: ", std_reward)
    env.close()