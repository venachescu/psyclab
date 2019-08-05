import numpy as np


def min_jerk_trajectory(start=0.0, goal=1.0, duration=1.0, time_step=0.01):

    n_time_steps = int(round(duration / time_step))
    x = np.zeros(n_time_steps)
    xd = np.zeros(n_time_steps)
    xdd = np.zeros(n_time_steps)

    x[0] = start

    for i, t in enumerate(np.linspace(0.0, duration, n_time_steps-1)):
        y, yd, ydd = min_jerk_step(x[i], xd[i], xdd[i], goal, duration-t, time_step)
        x[i+1], xd[i+1], xdd[i+1] = y, yd, ydd

    return x, xd, xdd


def min_jerk_step(x, xd, xdd, goal, tau, dt):

    if tau < dt:
        return x, xd, xd

    dist = goal - x

    a1 = 0
    a0 = xdd * tau ** 2.0
    v1 = 0
    v0 = xd * tau

    t1 = dt
    t2 = dt**2
    t3 = dt**3
    t4 = dt**4
    t5 = dt**5

    c1 = (6.*dist + (a1 - a0)/2. - 3.*(v0 + v1))/tau**5
    c2 = (-15.*dist + (3.*a0 - 2.*a1)/2. + 8.*v0 + 7.*v1)/tau**4
    c3 = (10.*dist + (a1 - 3.*a0)/2. - 6.*v0 - 4.*v1)/tau**3
    c4 = xdd/2.
    c5 = xd
    c6 = x

    x = c1*t5 + c2*t4 + c3*t3 + c4*t2 + c5*t1 + c6
    xd = 5.*c1*t4 + 4*c2*t3 + 3*c3*t2 + 2*c4*t1 + c5
    xdd = 20.*c1*t3 + 12.*c2*t2 + 6.*c3*t1 + 2.*c4

    return x, xd, xdd


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    data = min_jerk_trajectory(start=-1.0, goal=2.0, duration=5.0, time_step=0.0001)
    fig, axs = plt.subplots(3)
    [ax.plot(x) for x, ax in zip(data, axs)]
    fig.show()

    print('The jerk is {0}, but minimal, I promise.'.format(np.linalg.norm(np.diff(data[2]))))
