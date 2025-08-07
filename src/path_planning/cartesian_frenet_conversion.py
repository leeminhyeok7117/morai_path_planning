#!/usr/bin/env python
# -*-coding:utf-8-*-
"""
Created on Thu Aug  6 16:58:05 2020

@author: Elitebook 8570w
"""
import numpy as np
from math import sin, cos, tan, copysign, sqrt, degrees, pi
from scipy.spatial import distance
import os, sys

mission_name = None
initialize_mission_flag = False
last_closestrefpoint = 0


def get_mission_coord():
    global mission_name

    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    # from state_fmtc import mission_coord

    mission_name = ""
    return mission_name


def getClosestSPoint(rx, ry, x, y, last_search, iteration, mode, mission):
    global initialize_mission_flag
    global mission_name
    global last_closestrefpoint

    if initialize_mission_flag is False:
        get_mission_coord()
        initialize_mission_flag = True
        
    closestrefpoint, low, upper = 0, 0, 0
    position = np.array([x, y])

    if mode == 0:
        searchFlag = (last_search + iteration) < len(rx) and (last_search - iteration) > 0
        low = last_search - iteration if searchFlag else 0
        upper = last_search + iteration if searchFlag else len(rx)

        ref_positions = np.column_stack((rx[low:upper], ry[low:upper]))
        distances = np.linalg.norm(ref_positions - position, axis=1)
        try:
            closestrefpoint = np.argmin(distances) + low
        except ValueError:
            closestrefpoint = last_closestrefpoint

    elif mode == 1:
        searchFlag = (last_search + iteration) < len(rx) and (last_search - iteration) > 0

        if mission is None:
            low = last_search - iteration if searchFlag else 0
            upper = last_search + iteration if searchFlag else len(rx)
        else:
            low = last_search - iteration if searchFlag else int(max(mission_name[mission][0] * 10 - 20, 0))
            upper = last_search + iteration if searchFlag else int(min(mission_name[mission][1] * 10 + 20, len(rx)))

        ref_positions = np.column_stack((rx[low:upper], ry[low:upper]))
        distances = np.linalg.norm(ref_positions - position, axis=1)

        try:
            closestrefpoint = np.argmin(distances) + low
            last_closestrefpoint = closestrefpoint
        except ValueError:
            closestrefpoint = last_closestrefpoint
    else:
        pass

    return closestrefpoint

def calcOffsetPoint(x, y, cur_rx, cur_ry, cur_ryaw):
    position = np.array([x, y])
    ref_point = np.array([cur_rx, cur_ry])
    base_vec = np.array([cos(cur_ryaw), sin(cur_ryaw)])
    pose_vec = np.array(position - ref_point)

    return copysign(distance.euclidean(position, ref_point), np.cross(base_vec, pose_vec))

def sl2xy(s, l, cur_rx, cur_ry, cur_ryaw):
    x = cur_rx + l*cos(cur_ryaw + pi/2)
    y = cur_ry + l*sin(cur_ryaw + pi/2)

    return x, y

def main():
    pass

if __name__ == '__main__':
    main()