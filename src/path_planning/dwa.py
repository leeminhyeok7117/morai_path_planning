#!/usr/bin/env python3
# -*-coding:utf-8-*-

import rospy
import time

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Point32
from std_msgs.msg import Float64, Header
import numpy as np
import os, sys

from global_path import GlobalPath

"""
mode 0: 차선 변경 없이 한 차선 안에서 좁은 범위의 정적 장애물 회피
mode 1: 차선을 변경 해야 하는 넓은 범위의 정적 장애물 회피 (왼쪽 차선 -> 오른쪽 차선)
mode 2: gps mapping 을 이용한 track 에서 rubber cone 회피
mode 3: 유턴
"""
W_GLOBAL_OFFSET = 10.0 #safety cost 가중치
W_CONSISTENCY = 0.0 #smoothness cost 가중치
W_OBSTACLE = 1.5
W_MIDDLE_OBSTACLE = 2.0
COLLISION_COST = 0.5 #m

class DWA:
    def __init__(self, gp_name):
        self.candidate_pub = rospy.Publisher('/CDpath_dwa', PointCloud2, queue_size=10)
        self.selected_pub = rospy.Publisher('/SLpath_dwa', PointCloud2, queue_size=10)

        self.current_speed_sub = rospy.Subscriber('/speed', Float64, self.current_speed_callback, queue_size=1)
        self.obstacle_sub = rospy.Subscriber('/global_obs', PointCloud2, self.lidar_callback, queue_size=10)

        self.glob_path = GlobalPath(gp_name)

        self.visual = True
        self.cd_path = None
        self.sel_path = None

        # 로봇의 운동학적 모델 상수 설정
        self.max_speed = 50.0  # 최고 속도 [m/s]
        self.max_steer = np.deg2rad(37.0)  # 37도 [deg]
        self.possible_steer = 0.0 # 현재 속도에 따라 조정되는 steer 각도
        self.max_a = 1.0  # 내가 정하면 됨 [m/s^2]

        self.length = 4.635  # 차 길이 [m]
        self.width = 1.892  # 차 폭 [m]
        self.wheel_base = 3.00  # 차축 간 거리 [m]

        self.predict_time = 0.12  # 미래의 위치를 위한 예측 시간
        self.search_frame = 5  # 정수로 입력 (range 에 사용)
        self.DWA_search_size = [3, 7]  # Dynamic Window 에서 steer 의 분할 수 (홀수 and 정수)
        self.obstacle_force = 2.0  # 2m

        self.last_selected_path = []
        self.processed = []

        self.max_cost = 100

        self.frame_param = 0.5

    ######################################################

    def visual_selected_path(self, selected_path):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "glob_path"

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        structured_dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
        points = np.array([(p[0], p[1], 0.0) for p in selected_path], dtype=structured_dtype)

        pc2_msg = point_cloud2.create_cloud(header=header, fields=fields, points=points)
        pc2_msg.is_dense = True 
                
        self.selected_pub.publish(pc2_msg)
    
    def visual_candidate_paths(self, candidate_paths):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "glob_path"
    
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        structured_dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
        points = np.array([(p[0], p[1], 0.0) for path in candidate_paths for p in path], dtype=structured_dtype)
                
        pc2_msg = point_cloud2.create_cloud(header=header, fields=fields, points=points)
        pc2_msg.is_dense = True 

        self.candidate_pub.publish(pc2_msg)

    ######################################################

    def current_speed_callback(self, data):
        max_steer = 37.0
        min_steer = 10.0
        self.current_speed = data.data
        self.search_frame = max(int(5 + (self.current_speed - 25) * (5 / 25)), 5)
        if self.current_speed <= 25:
            self.possible_steer = max_steer
            self.search_frame = 5
        elif self.current_speed >25 and self.current_speed <35:
            self.possible_steer = min_steer
            # ratio = (self.current_speed - 25) / (35 - 25)  # 0~1
            # self.possible_steer = max_steer - ratio * (max_steer - min_steer)
            self.search_frame = 5
        else:
            self.possible_steer = min_steer
            self.search_frame = 5

    def lidar_callback(self, data):  
        raw_points = point_cloud2.read_points(
            data,
            field_names=("x","y","z","intensity"),
            skip_nans=True
        )

        points_list = []
        for p in raw_points:
            points_list.append([p[0], p[1], p[2], p[3]])
        
        if not points_list:
            self.processed = None
            return

        points = np.array(points_list, dtype=np.float32)
        
        if points.ndim == 1:
            points = points[np.newaxis, :]

        self.processed = points[:, :2]  

    ######################################################

    def convert_coordinate_l2g(self, d_x, d_y, d_theta): 
        d_theta = -np.pi / 2 + d_theta
        x_global = np.cos(d_theta) * d_x - np.sin(d_theta) * d_y
        y_global = np.sin(d_theta) * d_x + np.cos(d_theta) * d_y
        theta_global = np.pi /2 + d_theta
        return np.stack([x_global, y_global, theta_global], axis=1)
        
    def generate_predict_point(self, x, y, velocity, steer, heading):  
        # 접선 이동 거리 (= 호의 길이로 사용할 예정, 각 점마다의 거리임)
        tan_dis = np.clip(velocity * self.predict_time / 3.6 , 0.6, 1.0)
        
        # Assuming Bicycle model, (곡률 반경) = (차축 간 거리) / tan(조향각)
        R = self.wheel_base / np.tan(-steer) if steer != 0.0 else float('inf')
        theta_arr, future_pos = [], []
        frame_arr = np.arange(self.search_frame)
        
        if np.isinf(R):
            theta_arr = np.zeros_like(frame_arr, dtype=float)
            dx = np.zeros_like(frame_arr, dtype=float)
            dy = tan_dis * (frame_arr + 1)
        else:
            theta_arr = np.cumsum(np.full(self.search_frame, tan_dis / R))
            dx = R * (1 - np.cos(theta_arr))
            dy = R * np.sin(theta_arr)

        future_pos = self.convert_coordinate_l2g(dx, dy, theta_arr + heading)
        future_pos[:, 0] += x
        future_pos[:, 1] += y
        return future_pos  # return 값은 global coordinate 의 예측 점 x, y 좌표  -> [[x1, y1, theta1], [x2, y2, theta2], .....]

    def calc_dynamic_window(self): #steer=0.0고정
        DWA_steer = np.linspace(-self.possible_steer, self.possible_steer, self.DWA_search_size[1])
        DWA_velocity = self.current_speed + self.max_a
        dw = [DWA_velocity, DWA_steer]
        return dw

    def calc_global_cost(self, pose, target_index):
        target_x, target_y = self.glob_path.rx[target_index], self.glob_path.ry[target_index]
        cost = np.hypot(target_x - pose[-1, 0], target_y - pose[-1, 1])
        if cost > 3.0:
            cost = self.max_cost
        return cost
    
    def calc_obstacle_cost(self, x_arr, y_arr):
        path_xy = np.stack((x_arr, y_arr), axis=1)
        obs_xy = self.processed
        if self.processed is None or len(self.processed) == 0:
            return 0.0
        
        dists = np.linalg.norm(path_xy[None, :, :] - obs_xy[:, None, :], axis=2)
        min_dists = np.min(dists, axis=0)

        obstacle_cost = np.sum(np.where(min_dists <= COLLISION_COST, 1e6, 1.0 / (min_dists + 1e-6)))
        return obstacle_cost
    
    def calc_middle_obstacle_cost(self, x_arr, y_arr):
        path_xy = np.array([x_arr[len(x_arr)//4], y_arr[len(y_arr)//4]])
        obs_xy = self.processed
        if self.processed is None or len(self.processed) == 0:
            return 0.0
        
        dists = np.linalg.norm(obs_xy - path_xy, axis=1)
        min_dists = np.min(dists, axis=0)

        obstacle_cost = np.sum(np.where(min_dists <= COLLISION_COST, 1e6, 1.0 / (min_dists + 1e-6)))
        return obstacle_cost
    
    def calc_consistency_cost(self, future_pos, last_selected_path):
        if len(last_selected_path) == 0:
            return 0
        if not isinstance(last_selected_path, np.ndarray) or last_selected_path.ndim != 2:
            last_selected_path = np.array(last_selected_path)
        min_len = min(len(future_pos), len(last_selected_path))
        diff = np.abs(future_pos[:min_len, 2] - last_selected_path[:min_len, 2])
        consistency_cost = np.sum(diff) / len(last_selected_path)
        return consistency_cost
                
    def DWA(self, x, y, heading, current_s): 
        best_cost = float('inf')
        candidate_paths, selected_path = [], [] 

        dw = self.calc_dynamic_window()
        velocity = dw[0]

        path_dis = (self.current_speed * self.predict_time / 3.6 + 0.1) * 5
        if path_dis > 5: path_dis = 5
        index_offset = path_dis * 10
        target_index = int(current_s + index_offset)

        for steer in dw[1]:
            future_pos = self.generate_predict_point(x, y, velocity, steer, heading)
            
            candidate_paths.append(future_pos)
    
            cost = W_GLOBAL_OFFSET * self.calc_global_cost(future_pos, target_index) + \
                    W_OBSTACLE * self.calc_obstacle_cost(future_pos[:, 0], future_pos[:, 1]) + \
                    W_MIDDLE_OBSTACLE * self.calc_middle_obstacle_cost(future_pos[:, 0], future_pos[:, 1]) + \
                    W_CONSISTENCY * self.calc_consistency_cost(future_pos, self.last_selected_path)
            
            if cost < best_cost:
                best_cost = cost
                selected_path = future_pos
                self.last_selected_path = selected_path

        if self.visual:
            self.visual_candidate_paths(candidate_paths)
            self.visual_selected_path(selected_path)

        return selected_path
    
