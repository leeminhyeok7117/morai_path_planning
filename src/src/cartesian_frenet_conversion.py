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


def getClosestSPoint(rx, ry, x, y, last_search, iteration, mode, mission):  # 연산 속도 높이기 성공
    global initialize_mission_flag
    global mission_name
    global last_closestrefpoint

    if initialize_mission_flag is False:
        get_mission_coord()
        initialize_mission_flag = True

    closestrefpoint, low, upper = 0, 0, 0

    if mode == 0:  # 가장 가까운 s 인덱스를 찾을 때 기존 위치 근처에서부터 탐색
        searchFlag = (last_search + iteration) < len(rx) and (last_search - iteration) > 0
        low = last_search - iteration if searchFlag else 0
        upper = last_search + iteration if searchFlag else len(rx)

        mindistance = 999
        position = [x, y]
        closestrefpoint = 0

        for i in range(low, upper-1):
            ref_position = [rx[i], ry[i]]
            t_distance = distance.euclidean(position, ref_position)
            if t_distance < mindistance:
                mindistance = t_distance
                closestrefpoint = i
            else:
                continue

    elif mode == 1:  # 가장 가까운 s 인덱스를 찾을 때 전체 경로에서 탐색
        position = np.array([x, y])
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


""""
def cartesian_to_frenet(self,cur_rs,cur_rx,cur_ry,rtheta,rkappa,rdkappa,x,y,v,a,theta,kappa):
    dx = x - self.rx
    dy = y - self.ry
    d, d_d, d_dd, s, s_d, s_dd = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    cos_theta_r = cos(self.ryaw)
    sin_theta_r = sin(self.ryaw)

    cross_rn_nd = cos_theta_r * dy - sin_theta_r * dx
    d = copysign(sqrt(dx*dx+dy*dy), cross_rn_nd)

    delta_theta = theta - self.ryaw
    tan_delta_theta = tan(delta_theta)
    cos_delta_theta = cos(delta_theta)

    one_minus_kappa_r_d = 1 - self.rk * d
    d_d = one_minus_kappa_r_d * tan_delta_theta

    kappa_r_d_prime = self.rdk * d + self.rk * d_d
    d_dd = -1.0 * kappa_r_d_prime * tan_delta_theta + one_minus_kappa_r_d / cos_delta_theta / cos_delta_theta * (kappa * one_minus_kappa_r_d/ cos_delta_theta - self.rk)

    s = rs

    s_d = v * cos_delta_theta/ one_minus_kappa_r_d
    delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * kappa - self.rk

    s_dd = (a * cos_delta_theta - s_d * s_d *
            (d_d * delta_theta_prime - kappa_r_d_prime))/one_minus_kappa_r_d

    return s, s_d, s_dd, d, d_d, d_dd
"""


def main():
    """
    import rospy
    import matplotlib.pyplot as plt
    import sensor.sensor_data_communication as sensor_data_communication

    rospy.init_node("test_converter")
    dataHub = sensor_data_communication.sensor_data_communicationer()

    testPath = path_data.Path()

    x, y, yaw = [], [], []
    a = True
    while not rospy.is_shutdown():
        while not dataHub.readySensor():
            print("Sensor is not ready")
            continue
        if a:
            print("start!")
            a = False
        x, y, yaw = dataHub.get_pose()
        #refpoint = converter.getClosestSPoint(x, y)
        s, l = testPath.xy2sl(x,y)
        print("s, l", s, l)
        re_x, re_y = testPath.sl2xy(s, l)
        print("x, y", re_x, re_y)
        rate.sleep()
    """


if __name__ == '__main__':
    main()
