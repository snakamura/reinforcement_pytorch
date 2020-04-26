import numpy as np
import matplotlib.pyplot as plt
import agent
from maze import Maze

if __name__ == '__main__':
    pi_0 = agent.simple_convert_theta_to_pi(agent.THETA_0)
    state_action_history = agent.move_to_goal(pi_0)
    print(pi_0)
    print(state_action_history)

    maze = Maze(plt, agent.WIDTH, agent.HEIGHT, agent.WALLS, agent.START, agent.GOAL)
    anim = maze.animate_state_history([s for s, a in state_action_history])
    plt.show()
