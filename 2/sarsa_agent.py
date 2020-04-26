import numpy as np
import matplotlib.pyplot as plt
import agent
from q_base_agent import QBaseAgent
from maze import Maze

class SarsaAgent(QBaseAgent):
    def update_q(self, q, state, action, reward, next_state, next_action, eta, gamma):
        if next_state == agent.GOAL:
            q[state, action] += eta * (reward - q[state, action])
        else:
            q[state, action] += eta * (reward + gamma * q[next_state, next_action] - q[state, action])
        return q

if __name__ == '__main__':
    [states, actions] = agent.THETA_0.shape
    q_0 = np.random.rand(states, actions) * agent.THETA_0 * 0.1
    pi_0 = agent.simple_convert_theta_to_pi(agent.THETA_0)
    eta = 0.1
    gamma = 0.9
    epsilon = 0.5

    sarsaAgent = SarsaAgent(pi_0, eta, gamma, epsilon)
    q, state_action_history, state_values_history = sarsaAgent.solve(q_0)

    print(q)
    print(state_action_history)

    maze = Maze(plt, agent.WIDTH, agent.HEIGHT, agent.WALLS, agent.START, agent.GOAL)
#    anim = maze.animate_state_history([s for s, a in state_action_history])
    anim = maze.animate_state_values_history(state_values_history)
    plt.show()
