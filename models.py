import torch as th
from torch import nn
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size = 8, stride = 4, padding = 0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2, padding = 0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
    
from torchvision.models import resnet18, ResNet18_Weights

class CustomResNet18(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):

        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.net = resnet18(weights=ResNet18_Weights)
        self.net.conv1 = nn.Sequential(
                            nn.Conv2d(n_input_channels, 64, kernel_size = 7, stride = 2, padding = 3),
                            nn.BatchNorm2d(64),
                            nn.ReLU())
        self.net.fc = nn.Linear(in_features=512, out_features=128)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)