from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_device


def init_model(params, hyperparams):
    model = None
    model = PPO(**params, **hyperparams, device=get_device())
    return model


def get_model(model_path):
    model = None
    model = PPO.load(model_path)
    return model
