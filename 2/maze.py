import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

class Maze:
    def __init__(self, plt, width, height, walls, start, goal):
        self.__width = width
        self.__height = height
        self.__start = start
        self.__goal = goal

        self.__fig = plt.figure(figsize=(5, 5))
        self.__axes = self.__fig.gca()

        axes = self.__axes

        for s, e in walls:
            axes.plot(s, e, color='red', linewidth=2)

        for state in range(width * height):
            x, y = self.__position(state)
            axes.text(x, y, f'S{state}', size=14, ha='center')

        startX, startY = self.__position(start)
        axes.text(startX, startY - 0.2, 'START', ha='center')

        goalX, goalY = self.__position(goal)
        axes.text(goalX, goalY - 0.2, 'GOAL', ha='center')

        axes.set_xlim(0, width)
        axes.set_ylim(0, height)
        plt.tick_params(axis='both', which='both', top=False, bottom=False,
            left=False, right=False, labelbottom=False, labelleft=False)

        self.__current, = axes.plot([startX], [startY], marker='o', color='g', markersize=60)

    def __position(self, state):
        y, x = divmod(state, self.__width)
        return (x + 0.5, 2.5 - y)

    def animate_state_history(self, state_history):
        def init():
            self.__current.set_data([], [])

        def animate(step):
            x, y = self.__position(state_history[step])
            self.__current.set_data(x, y)

        return animation.FuncAnimation(self.__fig, animate, init_func=init,
            frames=len(state_history), interval=200, repeat=False)

    def animate_state_values_history(self, state_values_history):
        def init():
            self.__current.set_data([], [])

        def animate(step):
            for state in range(0, self.__width * self.__height):
                x, y = self.__position(state)
                state_value = 1.0 if state == self.__goal else state_values_history[step][state]
                self.__axes.plot([x], [y], marker='s', color=cm.jet(state_value), markersize=85)

        return animation.FuncAnimation(self.__fig, animate, init_func=init,
            frames=len(state_values_history), interval=200, repeat=False)

if __name__ == '__main__':
    walls = [
        ([1, 1], [0, 1]),
        ([1, 2], [2, 2]),
        ([2, 2], [2, 1]),
        ([2, 3], [1, 1]),
    ]

    maze = Maze(plt, 3, 3, walls, 0, 8)
    anim = maze.animate_state_history([0, 3, 4, 7, 8])

    plt.show()
