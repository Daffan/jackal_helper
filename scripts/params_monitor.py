import rospy
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

param_list = [
    'max_vel_x',
    'max_vel_theta',
    'vx_samples',
    'vtheta_samples',
    'path_distance_bias',
    'goal_distance_bias'
]

if __name__ == "__main__":

    fig, axe = plt.subplots(6, 1, figsize = (6, 18))
    xs = 0
    ys = [[0.7], [1.57], [6], [20], [0.75], [1]]

    def animate(i, xs, ys):
        for j, pn in enumerate(param_list):
            ys[j].append(rospy.get_param('/move_base/TrajectoryPlannerROS/' + pn))
            ys[j] = ys[j][-20:]
        # Draw x and y lists
        for j in range(6):
            axe[j].clear()
            axe[j].plot(ys[j])
            axe[j].set_ylabel(param_list[j])

    ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=100)
    plt.show()
