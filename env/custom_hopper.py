"""Implementation of the Hopper environment supporting
domain randomization optimization."""
import csv
import pdb
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv
from scipy.stats import truncnorm


class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None):
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        self.coeff_gravity = None
        self.coeff_thick = None
        self.coeff_friction = None
        self.coeff_masses = None
        self.udr_training = False
        self.extended_udr_training = False

        self.original_masses = np.copy(self.sim.model.body_mass[1:])  # Default link masses
        self.udr_dim = self.original_masses.shape[0] - 1  # Number of links
        self.min_udr = np.zeros(self.udr_dim)  # Lower bound of domain randomization distribution
        self.max_udr = np.zeros(self.udr_dim)  # Upper bound of domain randomization distribution

        self.original_geom_friction = np.copy(self.sim.model.geom_friction)
        self.original_geom_size = np.copy(self.sim.model.geom_size)
        self.original_gravity = np.copy(self.sim.model.opt.gravity[2])

        self.stats_masses = []
        self.stats_friction = []
        self.stats_gravity = []
        self.stats_thickness = []

        if domain == 'source':  # Source environment has an imprecise torso mass (1kg shift)
            self.sim.model.body_mass[1] -= 1.0

    def record_stats(self):
        self.stats_masses.append(np.copy(self.sim.model.body_mass[2:]))
        if self.extended_udr_training:
            self.stats_friction.append(np.copy(self.sim.model.geom_friction))
            self.stats_gravity.append(np.copy(self.sim.model.opt.gravity[2]))
            self.stats_thickness.append(np.copy(self.sim.model.geom_size))

    def get_stats(self):
        return self.stats_masses, self.stats_friction, self.stats_gravity, self.stats_thickness

    def set_udr_training(self, flag):
        self.udr_training = flag

    def set_extended_udr_training(self, flag):
        self.extended_udr_training = flag

    def get_udr_training(self):
        return self.udr_training

    def set_random_parameters(self):
        self.set_parameters(*self.sample_parameters())
        return

    def set_udr_distribution(self, bounds):
        for i in range(len(bounds) // 2):
            self.min_udr[i] = bounds[2 * i]
            self.max_udr[i] = bounds[2 * i + 1]
        return

    def set_extended_udr_coefficient(self, coeff_masses, coeff_friction, coeff_thick, coeff_gravity):
        self.coeff_masses = coeff_masses
        self.coeff_friction = coeff_friction
        self.coeff_thick = coeff_thick
        self.coeff_gravity = coeff_gravity

    def set_random_extended_parameters(self):

        # masses
        self.sim.model.body_mass[2] = np.random.uniform(self.sim.model.body_mass[2] * (1 - 0.5 * self.coeff_masses),
                                                        self.sim.model.body_mass[2] * (1 + 0.5 * self.coeff_masses))
        self.sim.model.body_mass[3] = np.random.uniform(self.sim.model.body_mass[3] * (1 - 0.5 * self.coeff_masses),
                                                        self.sim.model.body_mass[3] * (1 + 0.5 * self.coeff_masses))
        self.sim.model.body_mass[4] = np.random.uniform(self.sim.model.body_mass[4] * (1 - 0.5 * self.coeff_masses),
                                                        self.sim.model.body_mass[4] * (1 + 0.5 * self.coeff_masses))

        # gravity
        self.sim.model.opt.gravity[2] = np.random.uniform(
            self.sim.model.opt.gravity[2] * (1 - 0.5 * self.coeff_gravity),
            self.sim.model.opt.gravity[2] * (1 + 0.5 * self.coeff_gravity))
        # thickness
        for i in range(1, len(self.sim.model.geom_size)):
            self.sim.model.geom_size[i, 0] = np.random.uniform(
                self.sim.model.geom_size[i, 0] * (1 - 0.5 * self.coeff_thick),
                self.sim.model.geom_size[i, 0] * (1 + 0.5 * self.coeff_thick))

        # friction
        for i in range(1, len(self.sim.model.geom_friction)):
            self.sim.model.geom_friction[i, 0] = np.random.uniform(
                self.sim.model.geom_friction[i, 0] * (1 - 0.5 * self.coeff_friction),
                self.sim.model.geom_friction[i, 0] * (1 + 0.5 * self.coeff_friction))

            self.sim.model.geom_friction[i, 1] = np.random.uniform(
                self.sim.model.geom_friction[i, 1] * (1 - 0.5 * self.coeff_friction),
                self.sim.model.geom_friction[i, 1] * (1 + 0.5 * self.coeff_friction))

            self.sim.model.geom_friction[i, 2] = np.random.uniform(
                self.sim.model.geom_friction[i, 2] * (1 - 0.5 * self.coeff_friction),
                self.sim.model.geom_friction[i, 2] * (1 + 0.5 * self.coeff_friction))

    def get_udr_distribution(self):
        return self.min_udr, self.max_udr

    def sample_parameters(self):
        return np.random.uniform(self.min_udr, self.max_udr, self.min_udr.shape)

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array(self.sim.model.body_mass[1:])
        return masses

    def set_parameters(self, *task):
        """Set each hopper link's mass to a new value"""
        self.sim.model.body_mass[2:] = task

    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    def reset_model(self):
        """Reset the environment to a random initial state"""
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        if self.udr_training:
            self.set_random_parameters()
        if self.extended_udr_training:
            self.set_random_extended_parameters()
        self.record_stats()
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


"""
    Registered environments
"""
gym.envs.register(
    id="CustomHopper-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
)

gym.envs.register(
    id="CustomHopper-source-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "source"}
)

gym.envs.register(
    id="CustomHopper-target-v0",
    entry_point="%s:CustomHopper" % __name__,
    max_episode_steps=500,
    kwargs={"domain": "target"}
)
