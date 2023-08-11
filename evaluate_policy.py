from stable_baselines3.common.evaluation import evaluate_policy as evaluator
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from stable_baselines3.common.monitor import Monitor
from policy_gradient import get_model
import gym

N_EVAL_EPISODES = 100

from gym.wrappers.pixel_observation import PixelObservationWrapper
from gym.wrappers.resize_observation import ResizeObservation
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from env.custom_hopper import *

def evaluate_policy(params):
    evaluate_domain = params['evaluate_domain']
    env = gym.make(f'CustomHopper-{evaluate_domain}-v0')
    env.seed(0)

    model = get_model(params['policy_gradient'], params['model_path'])
    mean_reward, std_reward = evaluator(env=env, model=model, n_eval_episodes=N_EVAL_EPISODES, render=False)
    print(mean_reward, std_reward)
    print(env.return_queue)
    reward_queue = env.return_queue
    length_queue = env.length_queue
    reward_episodes = []
    length_episodes = []
    while reward_queue:
        reward_episodes.append(reward_queue.pop())
    while length_queue:
        length_episodes.append(length_queue.pop())

    reward_episodes.reverse()
    length_episodes.reverse()
    avg_reward_episodes = []
    for i in range(N_EVAL_EPISODES):
        avg_reward_episodes.append(reward_episodes[i] / length_episodes[i])
    print(mean_reward, std_reward)
    return mean_reward, std_reward, avg_reward_episodes, reward_episodes
