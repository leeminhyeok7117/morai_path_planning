import time
from matplotlib import pyplot as plt
import numpy as np
import math

from cubic_hermit_planner import hermite_with_constraints
import optimize_with_do_mpc

def calc_states_list():
    x = np.arange(5.0, 20.0, 5)
    y = np.arange(-10.0, 12.0, 2)

    max_yaw = np.deg2rad(25.0)
    n = 10   #yaw값 개수
    step = (2 * max_yaw) / (n - 1)
    yaw = np.arange(-max_yaw, max_yaw+step, step)

    states = []
    for iyaw in yaw:
        for iy in y:
            for ix in x:
                states.append([ix, iy, iyaw])
    print("n_state:", len(states))

    return states


def save_lookup_table(file_name, table):
    np.savetxt(file_name, np.array(table),
               fmt='%s', delimiter=",", header="x,y,yaw,km,kf", comments="")

    print("lookup table file is saved as " + file_name)


def generate_lookup_table():
    states = calc_states_list()
    k0 = 0.0
    lookup_table = [[1.0, 0.0, 0.0, 0.0, 0.0]] #x, y, yaw, km, kf

    for state in states:
        x_ref, y_ref, yaw_ref,__ = hermite_with_constraints([0,0], [state[0],state[1]], 0, state[2])

        if x_ref is not None:
            print("find good path")
            x_end = round(x[-1], 6)
            y_end = round(y[-1], 6)
            yaw_end = yaw[-1]
            km = curvature[len(curvature)//2]
            kf = curvature[-1]
            lookup_table.append([x_end, y_end, yaw_end, km, kf])

    print("finish lookup table generation")
    save_lookup_table("lookup_table.csv", lookup_table)

    for state in lookup_table:
        state_plot = hermite_with_constraints([0,0], [state[0],state[1]], 0, state[2])
        x_c, y_c, yaw_c = state_plot[0] , state_plot[1], state_plot[2]
        plt.plot(x_c, y_c, "-r")

    plt.grid(True)
    plt.axis("equal")
    plt.show()

    print("Done")


def main():
    start = time.time()
    generate_lookup_table()
    end = time.time()
    print(end-start)

if __name__ == '__main__':
    main()