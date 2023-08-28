from gym import ObservationWrapper, spaces
import numpy as np

class ImageObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.observation_space['pixels']

    def observation(self, obs):
        return obs["pixels"]