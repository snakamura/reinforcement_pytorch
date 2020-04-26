import numpy as np
import matplotlib.pyplot as plt
import agent

class QBaseAgent:
    def __init__(self, pi, eta, gamma, initial_epsilon):
        self.__pi = pi
        self.__eta = eta
        self.__gamma = gamma
        self.__initial_epsilon = initial_epsilon

    def __get_action(self, state, q, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(agent.ACTIONS, p=self.__pi[state, :])
        else:
            return agent.ACTIONS[np.nanargmax(q[state, :])]

    def update_q(self, q, state, action, reward, next_state, next_action, eta, gamma):
        pass

    def __move_to_goal(self, q, epsilon):
        state = agent.START
        next_action = self.__get_action(state, q, epsilon)
        state_action_history = []

        while state != agent.GOAL:
            action = next_action
            state_action_history.append((state, action.index))
            next_state = action.move(state)
            if next_state == agent.GOAL:
                reward = 1
                next_action = agent.Action.NONE
            else:
                reward = 0
                next_action = self.__get_action(next_state, q, epsilon)

            q = self.update_q(q, state, action.index, reward, next_state, next_action.index, self.__eta, self.__gamma)

            state = next_state
        state_action_history.append((state, np.nan))

        return (q, state_action_history)

    def solve(self, q):
        epsilon = self.__initial_epsilon
        state_values = np.nanmax(q, axis=1)
        state_values_history = [state_values]

        episode = 0
        while episode < 100:
            print("Episode %d" % episode)

            epsilon /= 2
            q, state_action_history = self.__move_to_goal(q, epsilon)
            print(len(state_action_history))
            new_state_values = np.nanmax(q, axis=1)
            print(np.sum(np.abs(new_state_values - state_values)))
            state_values = new_state_values
            state_values_history.append(state_values)

            episode += 1

        return (q, state_action_history, state_values_history)
