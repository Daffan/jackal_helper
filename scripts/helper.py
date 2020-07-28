import matplotlib.pyplot as plt
import numpy as np

def visual_laserscan(ob, polar_goal = False):
    x_points = []
    y_points = []
    l = len(ob)
    theta = [(i+1-l/2)/l*2*2.3561899662 for i in range(l-3)]
    for i, p in enumerate(ob[:-3]):
        if p != 30:
            x_points.append(p*np.cos((i+1-l/2.)/l*2.*2.3561899662))
            y_points.append(p*np.sin((i+1-l/2.)/l*2.*2.3561899662))

    plt.figure(figsize=(6, 6))
    plt.scatter(x_points[:-1], y_points[:-1], c = 'b')
    # plt.polar(theta, ob[:-3])
    plt.scatter(0, 0, c = 'r')
    plt.scatter(ob[-3], ob[-2], c = 'y')
    plt.show()

    return

def obs_reduction(obs, reduction = 10, polar_goal = True):
    a = obs[:-3][::reduction]
    l = len(obs)
    if not polar_goal:
        a = np.concatenate([a, obs[-3:]])
    else:
        theta = (np.arctan(obs[-2]/obs[-3])/(2*2.3561899662)*l+l/2)//reduction
        a = np.concatenate([a, np.array([theta, obs[-1]])])
    return a
