import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import gym

def display(frames):
    frame = frames[0]
    plt.figure(figsize=(frame.shape[1] / 72.0, frame.shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frame)
    plt.axis('off')

    def animate(step):
        patch.set_data(frames[step])

    anim = animation.FuncAnimation(plt.gcf(), animate, len(frames), interval=50)

    anim.save('cartpole.mp4')
    plt.show()

ENV = 'CartPole-v0'
NUM_DIZITIED = 6
GAMMA = 0.99
ETA = 0.5
MAX_STEPS = 200
NUM_EPISODES = 1000

class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_q(self, observation, action, reward, observation_next):
        self.brain.update_q(observation, action, reward, observation_next)

    def get_action(self, observation, episode):
        return self.brain.decide_action(observation, episode)

class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.q = np.random.uniform(low=0, high=1, size=(NUM_DIZITIED**num_states, num_actions))

    def bins(self, min, max, num):
        return np.linspace(min, max, num + 1)[1:-1]

    def digitize_state(self, observation):
        cart_position, cart_velocity, pole_angle, pole_velocity = observation
        digitized = [
            np.digitize(cart_position, bins=self.bins(-2.4, 2.4, NUM_DIZITIED)),
            np.digitize(cart_velocity, bins=self.bins(-3.0, 3.0, NUM_DIZITIED)),
            np.digitize(pole_angle, bins=self.bins(-0.5, 0.5, NUM_DIZITIED)),
            np.digitize(pole_velocity, bins=self.bins(-2.0, 2.0, NUM_DIZITIED)),
        ]
        return sum([v * NUM_DIZITIED**n for n, v in enumerate(digitized)])

    def update_q(self, observation, action, reward, observation_next):
        state = self.digitize_state(observation)
        state_next = self.digitize_state(observation_next)
        max_q_next = max(self.q[state_next][:])
        self.q[state][action] += ETA * (reward + GAMMA * max_q_next - self.q[state, action])

    def decide_action(self, observation, episode):
        state = self.digitize_state(observation)
        epsilon = 0.5 * (1 / (episode + 1))
        if epsilon <= np.random.uniform(0, 1):
            action = np.argmax(self.q[state][:])
        else:
            action = np.random.choice(self.num_actions)
        return action

class Environment:
    def __init__(self):
        self.env = gym.make(ENV)
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        self.agent = Agent(num_states, num_actions)

    def run(self):
        complete_episodes = 0
        is_episode_final = False
        frames = []

        for episode in range(NUM_EPISODES):
            observation = self.env.reset()

            for step in range(MAX_STEPS):
                if is_episode_final:
                    frames.append(self.env.render(mode='rgb_array'))

                action = self.agent.get_action(observation, episode)
                observation_next, _, done, _ = self.env.step(action)

                if done:
                    if step < 195:
                        reward = -1
                        complete_episodes = 0
                    else:
                        reward = 1
                        complete_episodes += 1
                else:
                    reward = 0

                self.agent.update_q(observation, action, reward, observation_next)

                observation = observation_next

                if done:
                    print('{0} Episode: Finished after {1} time steps'.format(episode, step + 1))
                    break

            if is_episode_final:
                display(frames)
                break

            if complete_episodes > 10:
                print('Success')
                is_episode_final = True

env = Environment()
env.run()
