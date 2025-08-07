#!/usr/bin/env python
#-*-coding:utf-8-*-

import rospy
import time
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Point32
from std_msgs.msg import Float64, Header ,Int32

import numpy as np
import os, sys
import cvxpy as cp

from global_path import GlobalPath
import cubic_spline_planner

W_GLOB = 2.0  # 전역경로 추종 가중치
W_SMOOTH = 0.5  # 부드러움 가중치
W_JERK = 1.0  # jerk 가중치
W_OBS = 5.0  # 장애물 가중치

class FrenetOptimization:
    def __init__(self, gp_name):
        self.glob_path = GlobalPath(gp_name)
        self.path_pub = rospy.Publisher('/optimized_path', PointCloud2, queue_size=1)

        self.current_speed_sub = rospy.Subscriber('/speed', Float64, self.current_speed_callback, queue_size=1)
        self.obstacle_sub = rospy.Subscriber('/global_obs', PointCloud2, self.lidar_callback, queue_size=10)
        self.lane_judge_sub = rospy.Subscriber('/minhyeok', Int32, self.lane_judge_callback, queue_size=1)

        self.processed = []
        self.max_speed = 50.0
        self.current_speed = 0.0
        self.N = 10
        self.current_s = 0
        self.current_q = 0
        self.lane_judge = 0

        self.last_obstacle_update = rospy.Time.now()
        rospy.Timer(rospy.Duration(1.0), self.reset_obstacles)

    def visual_optimized_path(self, optimized_path):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "glob_path"

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        structured_dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
        points = np.array([(p[0], p[1], 0.0) for p in optimized_path], dtype=structured_dtype)

        pc2_msg = point_cloud2.create_cloud(header=header, fields=fields, points=points)
        pc2_msg.is_dense = True 
                
        self.path_pub.publish(pc2_msg)

    def current_speed_callback(self, data):
        self.current_speed = data.data

        if self.current_speed < 20.0:
            self.N = 10
        else:
            speed_over_20 = min(self.current_speed - 20.0, 30.0)
            scale = speed_over_20 / 30.0
            self.N = int(np.clip(10 + scale * 20, 10, 30))

    def lidar_callback(self, data):
        self.processed = [] 
        now = rospy.Time.now()

        raw_points = point_cloud2.read_points(
            data,
            field_names=("x", "y", "z", "intensity"),
            skip_nans=True
        )         
        cnt = 0

        for p in raw_points:
            x, y = float(p[0]), float(p[1])  
            try:
                s, q = self.glob_path.xy2sl(x, y)   
                self.processed.append([s, q])
                cnt += 1
            except Exception as e:
                rospy.logwarn_throttle(5.0, f"xy2sl 변환 오류: {e}")
                continue

        if cnt == 0:
            self.processed = None
        else:
            self.processed = np.asarray(self.processed, dtype=np.float32)

        self.last_obstacle_update = now

    def lane_judge_callback(self, data):
        self.lane_judge = data.data
    
    def reset_obstacles(self, event):
        if rospy.Time.now() - self.last_obstacle_update > rospy.Duration(3):
            self.processed = []

    def optimize_path_generate(self, current_s): 
        ds = 1.0
        s_list = current_s + np.array([i*ds for i in range(self.N)])
        q_list = cp.Variable(self.N)

        # 제약조건
        constraints = []
        # 레인 조건
        if self.lane_judge == -1:
            constraints.append(q_list >= 0)
        elif self.lane_judge == 1:
            constraints.append(q_list <= 0)

        # 전역경로 조건
        constraints.append(cp.abs(q_list) <= 4.0)

        # 목적함수
        objective = 0
        # 경로추종, 부드러움, jerk, 장애물 회피 
        global_path_function = cp.sum_squares(q_list)
        
        smoothness_cost = cp.sum_squares(q_list[1:] - q_list[:-1])

        jerk = []
        for i in range(self.N - 3):
            jerk.append(q_list[i+3] - 3*q_list[i+2] + 3*q_list[i+1] - q_list[i])
        jerk = cp.hstack(jerk)
        jerk_cost = cp.sum_squares(jerk)

        objective += W_GLOB * global_path_function + W_SMOOTH * smoothness_cost + W_JERK * jerk_cost

        if self.processed is not None and len(self.processed) > 0:

            safe_distance = 1.5
            for obstacle in self.processed:
                obs_s, obs_q = obstacle
                for i in range(self.N):
                    ds_val    = s_list[i] - obs_s
                    dq_expr   = q_list[i] - obs_q
                    dist_expr = ds_val**2 + cp.square(dq_expr)           # convex

                    # 슬랙 변수 (항상 0 이상)
                    p = cp.Variable(nonneg=True)

                    # ✅ DCP 형식: convex ≤ affine(=constant + p)
                    constraints.append(dist_expr <= safe_distance**2 + p)

                    objective += W_OBS * p
        
        problem = cp.Problem(cp.Minimize(objective), constraints)
        result = problem.solve(solver=cp.ECOS, max_iters=1000) 

        if q_list.value is None:
            rospy.logwarn("No valid path found")
            return s_list, None
        
        return s_list, q_list.value

    def optimized_path(self, current_s):
        s_list, q_list = self.optimize_path_generate(current_s)
        x_list, y_list = self.glob_path.sl2xy(s_list, q_list)

        self.visual_optimized_path(list(zip(x_list, y_list)))
        optimized_path = np.array(cubic_spline_planner.calc_spline_course(x_list, y_list, ds=0.1))

        return optimized_path