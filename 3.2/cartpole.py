import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import gym

frames = []
env = gym.make('CartPole-v0')
observation = env.reset()

for step in range(0, 200):
    frames.append(env.render(mode='rgb_array'))
    action = np.random.choice(2)
    observation, reward, done, info = env.step(action)

env.close()

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

display(frames)
