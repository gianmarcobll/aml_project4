from wrappers import *
#from gym.wrappers.pixel_observation import PixelObservationWrapper
# from wrappers import Grayscale
# from gym.wrappers.resize_observation import ResizeObservation
# from gym.wrappers.frame_stack import FrameStack
# from gym.wrappers.gray_scale_observation import GrayScaleObservation
from env.custom_hopper import *
from models import CNN
from policy_gradient import init_model
from evaluate_policy import evaluate_policy
from stable_baselines3 import PPO


def custom_extractor(model, n_features):
    
    """
    It will create custom feature extractor for cnn networks.
    
    :param model: feature extractor class that we want to use
    :param n_features: number of output features
    :type model: class
    :type n_features: int
    :return: kwargs of the model
    """
    
    kwargs = dict(
        features_extractor_class=model,
        features_extractor_kwargs=dict(features_dim=n_features)
    )
        
    return kwargs

N_TIMESTEPS = 100_000

def train_vision_based(params):
    source_domain = params['train_domain']

    env = gym.make(f"CustomHopper-{source_domain}-v0")
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

    policy_kwargs = dict(
        features_extractor_class=CNN, # to use other model: CustomCNN
        features_extractor_kwargs=dict(features_dim = 128), # was 128 in oldcnn
    )

    model = PPO("CnnPolicy", env, policy_kwargs = policy_kwargs, verbose = 1, batch_size = 32, learning_rate = 0.0003)
    model.learn(N_TIMESTEPS, progress_bar=True)
    # masses_distribution = env.get_stats()[0]

    # with open('masses_distribution.csv', 'w', newline='') as csvfile:
    #     my_writer = csv.writer(csvfile)
    #     my_writer.writerows(masses_distribution)

    model.save("./experiments/vision_based/model")

if __name__ == "__main__":
    print(gym.__version__)

    params = {
        'train_domain': 'source'
    }
    train_vision_based(params)
    print("Training finished")
    params = {
                'evaluate_domain': 'target',
                'model_path': "./experiments/vision_based/model/hopper.zip",
            }
    evaluate_policy(params=params)