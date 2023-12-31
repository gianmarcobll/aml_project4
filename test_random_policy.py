"""Test a random policy on the Gym Hopper environment

Play around with this code to get familiar with the
Hopper environment.

For example, what happens if you don't reset the environment
even after the episode is over?
When exactly is the episode over?
What is an action here?
"""
import gym
from env.custom_hopper import *

def main():
    render = False

    # env = gym.make('CustomHopper-source-v0')  # [2.53429174 3.92699082 2.71433605 5.0893801 ]
    # env = gym.make('CustomHopper-target-v0')  # [3.53429174 3.92699082 2.71433605 5.0893801 ] 
    env = gym.make('CustomHopper-source-v0')

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    print('Dynamics parameters:', env.get_parameters())  # masses of each link of the Hopper
    env.set_udr_distribution([0.9, 1.1, 1.9, 2.1, 2.9, 3.1])
    print(env.get_udr_distribution())
    env.set_udr_training(True)
    print(env.get_udr_training())

    n_episodes = 500

    for ep in range(n_episodes):  
        done = False
        state = env.reset()  # Reset environment to initial state
        print('Dynamics parameters:', env.get_parameters())

        while not done:  # Until the episode is over
            action = env.action_space.sample()  # Sample random action

            state, reward, done, info = env.step(action)  # Step the simulator to the next timestep

            #"""Step 4: vision-based
            #img_state = env.render(mode="rgb_array", width=224, height=224)
            #"""

            if render:
                env.render()


if __name__ == '__main__':
    main()