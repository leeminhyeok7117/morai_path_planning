#!/usr/bin/env python
#-*-coding:utf-8-*-

# 상암


import rospy
import time
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Point32
from std_msgs.msg import Float64, Header ,Int32
import numpy as np
import os, sys

from global_path import GlobalPath
import polynomial as polynomial
import frenet_path as frenet_path

W_OFFSET = 0.9 #safety cost 가중치 1.0
W_CONSISTENCY = 1.5 #smoothness cost 가중치 1.5
W_OBSTACLE = 3.0  #obstacle cost 가중치 2.0
W_MIDDLE_OBSTACLE = 1.5 #middle obstacle cost 가중치 0.0
W_LANE_JUDGE = 1000 # lane judge cost 가중치 1000
W_DIRECTION = 1000 # Direction cost 가중치 1000
COLLISION_COST = 0.5 #m

class TrajectoryPlanner: # path planner
    def __init__(self, gp_name):
        self.last_selected_path = frenet_path.Frenet_path() # for consistency cost
        self.glob_path = GlobalPath(gp_name)
        self.candidate_pub = rospy.Publisher('/CDpath_fp', PointCloud2, queue_size=1)
        self.selected_pub = rospy.Publisher('/SLpath_fp', PointCloud2, queue_size=1)

        self.current_speed_sub = rospy.Subscriber('/speed', Float64, self.current_speed_callback, queue_size=1)
        self.obstacle_sub = rospy.Subscriber('/global_obs', PointCloud2, self.lidar_callback, queue_size=10)
        self.lane_judge_sub = rospy.Subscriber('/minhyeok', Int32, self.lane_judge_callback, queue_size=1)
        self.visual = True

        self.processed = []
        self.current_speed = 0.0
        self.ds = 0.0
        self.current_s = 0
        self.current_q = 0
        self.lane_judge = 0
        self.fixed_dir = None     
        self.DIR_TH = 1.5

        self.S_MARGIN = 2    # 생성한 경로 끝 추가로 경로 따라서 생성할 길이

        self.last_obstacle_update = rospy.Time.now()
        rospy.Timer(rospy.Duration(1.0), self.reset_obstacles)

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
        self.current_speed = data.data
        ds_min, ds_max, v_max = 2.0, 20.0, 50.0
        self.ds = ds_min + (self.current_speed/v_max)*(ds_max - ds_min)

    def lidar_callback(self, data):  
        self.processed = []
        raw_points = point_cloud2.read_points(
            data,
            field_names=("x","y","z","intensity"),
            skip_nans=True
        )

        points_list = []
        for p in raw_points:
            points_list.append([p[0], p[1], p[2], p[3]])
        
        if not points_list:
            self.last_obstacle_update = rospy.Time.now()
            self.processed = None
            return

        points = np.array(points_list, dtype=np.float32)
        
        if points.ndim == 1:
            points = points[np.newaxis, :]

        self.processed = points[:, :2]  
        self.last_obstacle_update = rospy.Time.now()

    def lane_judge_callback(self, data):
        self.lane_judge = data.data
    
    def reset_obstacles(self, event):
        if rospy.Time.now() - self.last_obstacle_update > rospy.Duration(3):
            self.processed = []

    def generate_path(self, si, qi, dtheta): 
        # (si, qi): 시작상태, dtheta: heading - ryaw, ds: polynomial의 길이, qf: 종료상태 q
        eps = 1e-3
        # 각도 정규화(선택이지만 권장)
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        if abs(np.cos(dtheta)) < eps:
            dtheta = np.sign(dtheta) * (np.pi/2 - eps)

        # ds 최소값 보장
        self.ds = max(self.ds, 2.0)
        
        candidate_paths = []
        sl_d = 0.5
        sf_base = si + self.ds + self.S_MARGIN 

        qf_arr = np.linspace(-7.0, 7.0, 11)
        condition = [np.abs(qf_arr)<=0.1, (np.abs(qf_arr)>0.1)]
        choose = [sf_base+3.0, sf_base-1.0]
        sf_arr = np.select(condition, choose)

        for qf_i, sf_i in zip(qf_arr, sf_arr):
            sign_q = np.sign(qf_i)
            if self.fixed_dir is not None and sign_q not in (0, self.fixed_dir):
                continue
            fp = frenet_path.Frenet_path()
            qs = polynomial.cubic_polynomial(si, qi, dtheta, self.ds, qf_i)  
            fp.s = np.arange(si, sf_i, sl_d)
            fp.q = qs.calc_point(fp.s) 
     
            fp.x, fp.y = self.glob_path.sl2xy(fp.s, fp.q)
            fp.yaw = self.glob_path.get_current_reference_yaw()
            fp.k = qs.calc_kappa(fp.s, self.glob_path.get_current_reference_kappa())

            fp.offset_cost = abs(qf_i)
            fp.consistency_cost = self.calc_consistency_cost(fp.q, self.last_selected_path.q)
            fp.obstacle_cost = self.calc_obstacle_cost(fp.x, fp.y)
            fp.middle_obstacle_cost = self.calc_middle_obstacle_cost(fp.x, fp.y)
            fp.lane_judge_cost = self.calc_lane_judge_cost(qf_i)
            fp.total_cost = W_CONSISTENCY * fp.consistency_cost + \
                            W_OFFSET * fp.offset_cost + \
                            W_OBSTACLE * fp.obstacle_cost + \
                            W_MIDDLE_OBSTACLE * fp.middle_obstacle_cost + \
                            W_LANE_JUDGE * fp.lane_judge_cost
            
            candidate_paths.append(fp)

        return candidate_paths
    
    def calc_consistency_cost(self, target_q, last_selected_q):
        consistency_cost = 0
        min_len = min(len(target_q), len(last_selected_q))
        diff = np.abs(target_q[:min_len] - last_selected_q[:min_len])
        
        consistency_cost = np.sum(diff) / len(last_selected_q) if len(last_selected_q) > 0 else 0
        return consistency_cost
    
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
    
    def calc_lane_judge_cost(self, qf_i):
        if self.lane_judge == -1: # 왼쪽 차선
            return 1 if qf_i > 0 else 0
        if self.lane_judge == 1: # 오른쪽 차선
            return 1 if qf_i < 0 else 0
        else: # 중앙 차선
            # return 1 if qf_i < 0 else 0
            return 0
    
    def update_fixed_dir(self, end_q):
        if self.fixed_dir is None and abs(end_q) >= self.DIR_TH:
            self.fixed_dir = 1 if end_q > 0 else -1
        elif self.fixed_dir is not None and abs(end_q) < self.DIR_TH:
            self.fixed_dir = None

    def optimal_trajectory(self, x, y, heading):
        si, qi = self.glob_path.xy2sl(x, y)
        self.current_s = si
        self.current_q = qi

        ryaw = self.glob_path.get_current_reference_yaw_no_s()
        dtheta = heading - ryaw 
        
        candidate_paths = self.generate_path(si, qi, dtheta)

        selected_path = min(candidate_paths, key=lambda fp: fp.total_cost)
        self.last_selected_path = selected_path
        self.update_fixed_dir(selected_path.q[-1])

        ############### RVIZ 비쥬얼 코드 ##############
        if self.visual == True:
            self.visual_selected_path(list(zip(selected_path.x, selected_path.y)))
            self.visual_candidate_paths([list(zip(fp.x, fp.y)) for fp in candidate_paths])
        ##############################################
        
        return selected_path

# 원주

"""
import rospy
import time
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Point32
from std_msgs.msg import Float64, Header ,Int32
import numpy as np
import os, sys

from global_path import GlobalPath
import polynomial as polynomial
import frenet_path as frenet_path

W_OFFSET = 0.9 #safety cost 가중치 1.0
W_CONSISTENCY = 1.5 #smoothness cost 가중치 1.5
W_OBSTACLE = 0.1  #obstacle cost 가중치 2.0
W_MIDDLE_OBSTACLE = 0.0 #middle obstacle cost 가중치 0.0
W_LANE_JUDGE = 1000.0 # lane judge cost 가중치 1000.0
COLLISION_COST = 0.5 #m

class TrajectoryPlanner: # path planner
    def __init__(self, gp_name):
        self.last_selected_path = frenet_path.Frenet_path() # for consistency cost
        self.glob_path = GlobalPath(gp_name)
        self.candidate_pub = rospy.Publisher('/CDpath_fp', PointCloud2, queue_size=1)
        self.selected_pub = rospy.Publisher('/SLpath_fp', PointCloud2, queue_size=1)

        self.current_speed_sub = rospy.Subscriber('/speed', Float64, self.current_speed_callback, queue_size=1)
        self.obstacle_sub = rospy.Subscriber('/global_obs', PointCloud2, self.lidar_callback, queue_size=10)
        self.lane_judge_sub = rospy.Subscriber('/minhyeok', Int32, self.lane_judge_callback, queue_size=1)
        self.visual = True

        self.processed = []
        self.current_speed = 0.0
        self.ds = 0.0
        self.current_s = 0
        self.current_q = 0
        self.lane_judge = 0

        self.S_MARGIN = 1    # 생성한 경로 끝 추가로 경로 따라서 생성할 길이

        self.last_obstacle_update = rospy.Time.now()
        rospy.Timer(rospy.Duration(1.0), self.reset_obstacles)

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
        self.current_speed = data.data
        ds_min, ds_max, v_max = 2.0, 20.0, 50.0
        self.ds = ds_min + (self.current_speed/v_max)*(ds_max - ds_min)

    def lidar_callback(self, data):  
        self.processed = []
        raw_points = point_cloud2.read_points(
            data,
            field_names=("x","y","z","intensity"),
            skip_nans=True
        )

        points_list = []
        for p in raw_points:
            points_list.append([p[0], p[1], p[2], p[3]])
        
        if not points_list:
            self.last_obstacle_update = rospy.Time.now()
            self.processed = None
            return

        points = np.array(points_list, dtype=np.float32)
        
        if points.ndim == 1:
            points = points[np.newaxis, :]

        self.processed = points[:, :2]  
        self.last_obstacle_update = rospy.Time.now()

    def lane_judge_callback(self, data):
        self.lane_judge = data.data
    
    def reset_obstacles(self, event):
        if rospy.Time.now() - self.last_obstacle_update > rospy.Duration(3):
            self.processed = []

    def generate_path(self, si, qi, dtheta): 
        # (si, qi): 시작상태, dtheta: heading - ryaw, ds: polynomial의 길이, qf: 종료상태 q
        eps = 1e-3
        # 각도 정규화(선택이지만 권장)
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        if abs(np.cos(dtheta)) < eps:
            dtheta = np.sign(dtheta) * (np.pi/2 - eps)

        # ds 최소값 보장
        self.ds = max(self.ds, 2.0)
        
        candidate_paths = []
        sl_d = 0.5
        sf_base = si + self.ds + self.S_MARGIN 

        qf_arr = np.linspace(-2.5, 2.5, 11)
        condition = [np.abs(qf_arr)<=0.1, (np.abs(qf_arr)>0.1)]
        choose = [sf_base+3.0, sf_base-1.0]
        sf_arr = np.select(condition, choose)

        for qf_i, sf_i in zip(qf_arr, sf_arr):
            fp = frenet_path.Frenet_path()
            qs = polynomial.cubic_polynomial(si, qi, dtheta, self.ds, qf_i)  
            fp.s = np.arange(si, sf_i, sl_d)
            fp.q = qs.calc_point(fp.s) 
     
            fp.x, fp.y = self.glob_path.sl2xy(fp.s, fp.q)
            fp.yaw = self.glob_path.get_current_reference_yaw()
            fp.k = qs.calc_kappa(fp.s, self.glob_path.get_current_reference_kappa())

            fp.offset_cost = abs(qf_i)
            fp.consistency_cost = self.calc_consistency_cost(fp.q, self.last_selected_path.q)
            fp.obstacle_cost = self.calc_obstacle_cost(fp.x, fp.y)
            fp.middle_obstacle_cost = self.calc_middle_obstacle_cost(fp.x, fp.y)
            fp.lane_judge_cost = self.calc_lane_judge_cost(qf_i)
            fp.total_cost = W_CONSISTENCY * fp.consistency_cost + \
                            W_OFFSET * fp.offset_cost + \
                            W_OBSTACLE * fp.obstacle_cost + \
                            W_MIDDLE_OBSTACLE * fp.middle_obstacle_cost + \
                            W_LANE_JUDGE * fp.lane_judge_cost
            
            candidate_paths.append(fp)

        return candidate_paths
    
    def calc_consistency_cost(self, target_q, last_selected_q):
        consistency_cost = 0
        min_len = min(len(target_q), len(last_selected_q))
        diff = np.abs(target_q[:min_len] - last_selected_q[:min_len])
        
        consistency_cost = np.sum(diff) / len(last_selected_q) if len(last_selected_q) > 0 else 0
        return consistency_cost
    
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
    
    def calc_lane_judge_cost(self, qf_i):
        if self.lane_judge == -1: # 왼쪽 차선
            return 1 if qf_i > 0 else 0
        if self.lane_judge == 1: # 오른쪽 차선
            return 1 if qf_i < 0 else 0
        else: # 중앙 차선
            # return 1 if qf_i < 0 else 0
            return 0
        
    def optimal_trajectory(self, x, y, heading):
        si, qi = self.glob_path.xy2sl(x, y)
        self.current_s = si
        self.current_q = qi

        ryaw = self.glob_path.get_current_reference_yaw_no_s()
        dtheta = heading - ryaw 
        
        candidate_paths = self.generate_path(si, qi, dtheta)

        selected_path = min(candidate_paths, key=lambda fp: fp.total_cost)
        self.last_selected_path = selected_path

        ############### RVIZ 비쥬얼 코드 ##############
        if self.visual == True:
            self.visual_selected_path(list(zip(selected_path.x, selected_path.y)))
            self.visual_candidate_paths([list(zip(fp.x, fp.y)) for fp in candidate_paths])
        ##############################################
        
        return selected_path"""