#!/usr/bin/env python
# -*-coding:utf-8-*-

"""
Created on Mon Aug 24 15:12:36 2020

@author: JHP
"""

# 경로 데이터 로드, 좌표 변환, 

import numpy as np
import time
import bisect
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import cubic_spline_planner # 같은 path_planning 파일에 있어서 가능.
import cartesian_frenet_conversion # 얘도 마찬가지

MAX_SPEED = 3
MIN_SPEED = 1

MAX_CONERING_SPEED = 2
MIN_CONERING_SPEED = 1

# filename은 다른 파일에서 GlobalPath 객체를 생성해서 지정하면 덮어씌워짐.

class GlobalPath:
    def __init__(self, filename='/home/macaron/catkin_ws/src/macaron_5/path/round.npy', x=[], y=[], ds=0.1):
        # /home/macaron/catkin_ws/src/macaron_4/path/8jung_test2.npy
        self.rx, self.ry, self.ryaw, self.rk, self.rdk, self.s = [], [], [], [], [], 0.0
        if len(x) > 0:
            self.rx, self.ry, self.ryaw, self.rk, self.rdk, self.s = cubic_spline_planner.calc_spline_course(x, y, ds)
        else:
            pathArray = np.load(file=filename)
            gx = pathArray[0:pathArray.shape[0] - 1, 0]
            gy = pathArray[0:pathArray.shape[0] - 1, 1]
            self.rx, self.ry, self.ryaw, self.rk, self.rdk, self.s = cubic_spline_planner.calc_spline_course(gx, gy, ds)
        self.vel_param = 0
        self.rvel = self.det_target_speed()    
        self.cur_ref_index = 0
        self.cur_s_ref_index = 0
        self.last_search_time = 0
        self.last_local_index = 0

        self.rx_arr = np.array(self.rx)
        self.ry_arr = np.array(self.ry)
        self.ryaw_arr = np.array(self.ryaw)
        self.rk_arr = np.array(self.rk)
        self.s_arr = np.array(self.s)
   
    def getClosestSIndexCurXY(self, x, y, mode=0, base_iter=30, mission=None): 
        ref_index = 0

        if mode == 0:  # 가장 가까운 s 인덱스를 찾을 때 기존 위치 근처에서부터 탐색
            cur_time = time.time()
            time_elapsed = len(self.rx) if cur_time - self.last_search_time > len(
                self.rx) else cur_time - self.last_search_time
            iteration = int(time_elapsed + 1) * base_iter
            self.last_search_time = cur_time
            ref_index = cartesian_frenet_conversion.getClosestSPoint(self.rx, self.ry, x, y, self.cur_ref_index,
                                                                     iteration, mode=mode, mission=mission)
            self.cur_ref_index = ref_index

        elif mode == 1:  # 가장 가까운 s 인덱스를 찾을 때 전체 경로에서 탐색
            iteration = len(self.rx)
            ref_index = cartesian_frenet_conversion.getClosestSPoint(self.rx, self.ry, x, y, self.cur_ref_index,
                                                                     iteration, mode=mode, mission=mission)
        else:
            pass

        return ref_index                             

    def getClosestSIndexCurS(self, s):
        index = np.searchsorted(self.s_arr, s, side='right') - 1
        index = np.clip(index, 0 ,len(self.s_arr)-1) 
        return index    
    
    def xy2sl(self, x, y, mode=0, base_iter=30, mission=None):
        ref_index = self.getClosestSIndexCurXY(x, y, mode=mode, base_iter=base_iter, mission=mission)
        self.cur_ref_index = ref_index
        return self.s[ref_index], cartesian_frenet_conversion.calcOffsetPoint(x, y, self.rx[ref_index],
                                                                              self.ry[ref_index], self.ryaw[ref_index]) #l값
    
    def get_current_reference_yaw(self):
        yaw = self.ryaw_arr[self.cur_s_ref_index]
        return np.array(np.where(yaw<=0, yaw+2*np.pi, yaw))
    
    def get_current_reference_yaw_no_s(self):
        return self.ryaw_arr[self.cur_ref_index]

    def get_current_reference_kappa(self):
        return np.array(self.rk_arr[self.cur_s_ref_index])

    def sl2xy(self, s, l):
        ref_index = self.getClosestSIndexCurS(s)
        self.cur_s_ref_index = ref_index
        
        cur_rx = self.rx_arr[ref_index]
        cur_ry = self.ry_arr[ref_index]
        cur_ryaw = self.ryaw_arr[ref_index]
        
        x = cur_rx + l * np.cos(cur_ryaw + np.pi / 2)
        y = cur_ry + l * np.sin(cur_ryaw + np.pi / 2)
        
        return x, y

    def getPathFromTo(self, pos1, pos2):
        index1 = self.getClosestSIndexCurXY(pos1[0], pos1[1], 1)
        index2 = self.getClosestSIndexCurXY(pos2[0], pos2[1], 1)
        print(index1, index2)
        return self.rx[index1:index2], self.ry[index1:index2]
    
    def max_curvature(self, last_yaw, current_yaw):
       max_max = abs(current_yaw - last_yaw)
    
       if max_max >= np.pi:  # yaw 값이 360도를 넘어갈 때
           max_max = 2 * np.pi - max_max
       
       return max_max
    
    def det_target_speed(self):
        # 곡선도로 타겟 속도 계산
        max_index = len(self.ryaw)
        target_speed = MAX_SPEED
        target_speed_list = []
        for index in range(0, max_index):
            try:
                cur_diff_front = abs(self.max_curvature(self.ryaw[index+50], self.ryaw[index]))
                cur_diff_last = abs(self.max_curvature(self.ryaw[index+100], self.ryaw[index]))
                cur_diff = max(cur_diff_front, cur_diff_last)
            except:
                cur_diff = abs(self.max_curvature(self.ryaw[max_index-1], self.ryaw[index]))
                
            if cur_diff <= 5 * np.pi / 180:
                target_speed = int((MAX_CONERING_SPEED - MAX_SPEED)/(5 * np.pi/180) * cur_diff + MAX_SPEED)
            elif cur_diff >= 30 * np.pi / 180:
                target_speed = MIN_CONERING_SPEED
            else:
                target_speed = int(((MIN_CONERING_SPEED - MAX_CONERING_SPEED)/(25 * np.pi/180)) * (cur_diff - 5*np.pi/180) + MAX_CONERING_SPEED)
            
            # 직선도로 속도는 최대속도 (MAX_CONERING_SPEED --> MAX_SPEED)
            if cur_diff <= np.pi / 180:
                target_speed = MAX_SPEED    

            if target_speed >= MAX_SPEED:
                target_speed = MAX_SPEED
            elif target_speed <= MIN_SPEED:
                target_speed = MIN_SPEED
            
            target_speed_list.append(target_speed)
            index += 1
            
        return target_speed_list


def main():
    pass


if __name__ == '__main__':
    main()