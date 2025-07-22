#!/usr/bin/env python
# -*-coding:utf-8-*-

from matplotlib import pyplot as plt
import numpy as np

from cubic_hermite_planner import hermite_with_constraints

def calc_states_list():
    x = np.arange(3.0, 24.0, 3)
    y = np.arange(-10.0, 12.0, 2)

    max_yaw = np.deg2rad(25.0)
    n = 5   #yaw값 개수
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
        result = hermite_with_constraints([0,0], [state[0],state[1]], 0, state[2])

        ### 곡률, yaw값 제한 O
        if result is None:
            print("제한범위 넘어감")
            continue

        x, y, yaw, curvature = result
        if x is not None:
            print("find good path")
            x_end = round(x[-1], 4)
            y_end = round(y[-1], 4)
            yaw_end = round(yaw[-1], 4)
            km = round(curvature[len(curvature)//2], 4)
            kf = round(curvature[-1], 4)
            lookup_table.append([x_end, y_end, yaw_end, km, kf])
    print('실제 state 개수', len(x))
    print("finish lookup table generation")
    save_lookup_table("lookup_table.csv", lookup_table)

    #plot용 코드
    for state in lookup_table:
        state_plot = hermite_with_constraints([0,0], [state[0],state[1]], 0, state[2])
        x_c, y_c, yaw_c = state_plot[0] , state_plot[1], state_plot[2]
        plt.plot(x_c, y_c, "-r")

    plt.grid(True)
    plt.axis("equal")
    plt.show()

    print("Done")


def main():
    generate_lookup_table()

if __name__ == '__main__':
    main()