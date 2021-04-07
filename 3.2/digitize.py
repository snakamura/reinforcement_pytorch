import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import gym

ENV = 'CartPole-v0'
NUM_DIZITIED = 6

env = gym.make(ENV)
observation = env.reset()

def bins(min, max, num):
    return np.linspace(min, max, num + 1)[1:-1]

def digitize_state(observation):
    cart_position, cart_velocity, pole_angle, pole_velocity = observation
    digitized = [
        np.digitize(cart_position, bins=bins(-2.4, 2.4, NUM_DIZITIED)),
        np.digitize(cart_velocity, bins=bins(-3.0, 3.0, NUM_DIZITIED)),
        np.digitize(pole_angle, bins=bins(-0.5, 0.5, NUM_DIZITIED)),
        np.digitize(pole_velocity, bins=bins(-2.0, 2.0, NUM_DIZITIED)),
    ]
    return sum([v * NUM_DIZITIED**n for n, v in enumerate(digitized)])

print(digitize_state(observation))
