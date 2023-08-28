from wrappers import ImageObservationWrapper
from gym.wrappers.pixel_observation import PixelObservationWrapper
from gym.wrappers.resize_observation import ResizeObservation
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from env.custom_hopper import *
from models import CustomCNN
from models import CustomResNet18
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
import os

N_TIMESTEPS = 250_000

def vision_based_env(env):
    env = PixelObservationWrapper(env)
    print('state observation shape', env.observation_space.shape)
    env = ImageObservationWrapper(env)
    print('obervation shape after preprocessing:', env.observation_space.shape)
    env = ResizeObservation(env, shape=[224, 224])
    print('obervation shape after resizing:', env.observation_space.shape)        
    env = GrayScaleObservation(env, keep_dim=False)
    print('obervation shape after gray scaling:', env.observation_space.shape)
    env = FrameStack(env, num_stack=4)
    print('obervation shape after stacking:', env.observation_space.shape)
    return env

def train_vision_based(params):

    env = gym.make("CustomHopper-source-v0")
    print(env.observation_space)
    env = vision_based_env(env)
    target_env = gym.make('CustomHopper-target-v0')
    print(target_env.observation_space)
    target_env = vision_based_env(target_env)
    udr = ""
    if params['udr_enable']:
        masses_to_set = env.get_parameters()[1:]
        lower_bound = masses_to_set * (1 - 0.5 * 0.7)
        upper_bound = masses_to_set * (1 + 0.5 * 0.7)
        distribution_set = [lower_bound[0], upper_bound[0], lower_bound[1], upper_bound[1], lower_bound[2],
                            upper_bound[2]]
        env.set_udr_training(True)
        env.set_udr_distribution(distribution_set)
        udr = "_UDR"

    policy_kwargs = None
    lr = ''
    if params['learning_rate'] == 1e-3:
        lr = '0'
    else:
        lr = '1'

    if params['vision_based_model'] == "customCNN":
        policy_kwargs = dict(
            features_extractor_class=CustomCNN, # to use other model: CustomCNN
            features_extractor_kwargs=dict(features_dim = 128), # was 128 in oldcnn
        )
    elif params['vision_based_model'] == "resnet18":
        policy_kwargs = dict(
            features_extractor_class=CustomResNet18, # to use other model: CustomCNN
            features_extractor_kwargs=dict(features_dim = 128), # was 128 in oldcnn
        )
    if os.path.exists(f"./training/models/vision_based/{params['vision_based_model']}_LR_{lr}" + udr + ".zip"):
                    print("Found model!")
                    model = PPO.load(f"./training/models/vision_based/{params['vision_based_model']}_LR_{lr}" + udr, env=env)
    else:
        model = PPO("CnnPolicy", env, policy_kwargs = policy_kwargs, verbose = 1, learning_rate = params['learning_rate'])
        model.learn(N_TIMESTEPS, progress_bar=True)
        model.save(f"./training/models/vision_based/{params['vision_based_model']}_LR_{lr}" + udr)
    
    print("Source-Source environment results:")
    res = evaluate_policy(model, env, n_eval_episodes=50, render=False)
    print(res)
    print("Source-Target environment results:")
    res = evaluate_policy(model, target_env, n_eval_episodes=50, render=False)
    print(res)
    env.close()
    target_env.close()

if __name__ == "__main__":
    params = {
        'train_domain': 'source',
        'udr_enable': False,
        'vision_based_model': 'customCNN',
        'learning_rate': 1e-3
    }
    print("training with params = ", params)
    train_vision_based(params=params)
    params = {
        'train_domain': 'source',
        'udr_enable': False,
        'vision_based_model': 'customCNN',
        'learning_rate': 3e-4
    }
    print("training with params = ", params)
    train_vision_based(params=params)
    params = {
        'train_domain': 'source',
        'udr_enable': True,
        'vision_based_model': 'customCNN',
        'learning_rate': 1e-3
    }
    print("training with params = ", params)
    train_vision_based(params=params)
    params = {
        'train_domain': 'source',
        'udr_enable': True,
        'vision_based_model': 'customCNN',
        'learning_rate': 3e-4
    }
    print("training with params = ", params)
    train_vision_based(params=params)
    params = {
        'train_domain': 'source',
        'udr_enable': False,
        'vision_based_model': 'resnet18',
        'learning_rate': 1e-3
    }
    print("training with params = ", params)
    train_vision_based(params=params)
    params = {
        'train_domain': 'source',
        'udr_enable': False,
        'vision_based_model': 'resnet18',
        'learning_rate': 3e-4
    }
    print("training with params = ", params)
    train_vision_based(params=params)
    params = {
        'train_domain': 'source',
        'udr_enable': True,
        'vision_based_model': 'resnet18',
        'learning_rate': 1e-3
    }
    print("training with params = ", params)
    train_vision_based(params=params)
    params = {
        'train_domain': 'source',
        'udr_enable': True,
        'vision_based_model': 'resnet18',
        'learning_rate': 3e-4
    }
    print("training with params = ", params)
    train_vision_based(params=params)
    