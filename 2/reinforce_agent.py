import numpy as np
import matplotlib.pyplot as plt
import agent
from maze import Maze

def update_theta(theta, pi, state_action_history, eta):
    total_count = len(state_action_history)

    [states, actions] = theta.shape
    delta_theta = theta.copy()
    for state in range(states):
        all_actions = [sa for sa in state_action_history if sa[0] == state]
        all_count = len(all_actions)
        for action in range(actions):
            this_actions = [sa for sa in all_actions if sa[1] == action]
            this_count = len(this_actions)
            delta_theta[state, action] = (this_count - pi[state, action] * all_count) / total_count

    return theta + eta * delta_theta

def solve(theta, eta, stop_epsilon):
    while True:
        pi = agent.softmax_convert_theta_to_pi(theta)
        state_action_history = agent.move_to_goal(pi)
        theta = update_theta(theta, pi, state_action_history, eta)
        new_pi = agent.softmax_convert_theta_to_pi(theta)

        print(len(state_action_history), np.sum(np.abs(new_pi - pi)))

        if np.sum(np.abs(new_pi - pi)) < stop_epsilon:
            break

    return (pi, state_action_history)

if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)

    pi, state_action_history = solve(agent.THETA_0, 0.1, 10**-4)
    print(pi)
    print(state_action_history)

    maze = Maze(plt, agent.WIDTH, agent.HEIGHT, agent.WALLS, agent.START, agent.GOAL)
    anim = maze.animate_state_history([s for s, a in state_action_history])
    plt.show()
