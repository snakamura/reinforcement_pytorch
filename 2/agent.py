from enum import Enum
import numpy as np

WIDTH = 3
HEIGHT = 3
START = 0
GOAL = 8
WALLS = [
    ([1, 1], [0, 1]),
    ([1, 2], [2, 2]),
    ([2, 2], [2, 1]),
    ([2, 3], [1, 1]),
]

class Action(Enum):
    UP = (0, -3)
    RIGHT = (1, 1)
    DOWN = (2, 3)
    LEFT = (3, -1)
    NONE = (np.nan, 0)

    def __init__(self, index, move):
        self.index = index
        self._move = move

    def move(self, state):
        return state + self._move

ACTIONS=[Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]
THETA_0 = np.array([
    [np.nan, 1,      1,      np.nan],
    [np.nan, 1,      np.nan, 1     ],
    [np.nan, np.nan, 1,      1     ],
    [1,      1,      1,      np.nan],
    [np.nan, np.nan, 1,      1     ],
    [1,      np.nan, np.nan, np.nan],
    [1,      np.nan, np.nan, np.nan],
    [1,      1,      np.nan, np.nan],
])

def simple_convert_theta_to_pi(theta):
    [states, actions] = theta.shape
    pi = np.zeros((states, actions))
    for state in range(states):
        pi[state, :] = theta[state, :] / np.nansum(theta[state, :])
    pi = np.nan_to_num(pi)
    return pi

def softmax_convert_theta_to_pi(theta):
    beta = 1.0
    [states, actions] = theta.shape
    pi = np.zeros((states, actions))
    exp_theta = np.exp(beta * theta)
    for state in range(states):
        pi[state, :] = exp_theta[state, :] / np.nansum(exp_theta[state, :])
    pi = np.nan_to_num(pi)
    return pi

def get_action(state, pi):
    return np.random.choice(ACTIONS, p=pi[state, :])

def move_to_goal(pi):
    state = START
    state_action_history = []

    while state != GOAL:
        action = get_action(state, pi)
        state_action_history.append((state, action.index))
        state = action.move(state)
    state_action_history.append((state, np.nan))

    return state_action_history
