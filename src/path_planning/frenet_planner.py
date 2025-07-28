#!/usr/bin/env python
#-*-coding:utf-8-*-

import rospy
import time
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Point32
from std_msgs.msg import Float64, Header
import numpy as np
import os, sys

from global_path import GlobalPath
import polynomial as polynomial
import frenet_path as frenet_path

W_OFFSET = 1 #safety cost 가중치
W_CONSISTENCY = 0.5 #smoothness cost 가중치

class TrajectoryPlanner: # path planner
    def __init__(self, node, gp_name):
        self.last_selected_path = frenet_path.Frenet_path() # for consistency cost
        self.glob_path = GlobalPath(gp_name)
        self.node = node
        self.candidate_pub = rospy.Publisher('/CDpath_fp', PointCloud2, queue_size=10)
        self.selected_pub = rospy.Publisher('/SLpath_fp', PointCloud2, queue_size=10)

        self.visual = True

        self.current_speed = 0.0
        self.ds = 0.0
        self.current_s = 0
        self.current_q = 0

        self.S_MARGIN = 5    # 생성한 경로 끝 추가로 경로 따라서 생성할 길이

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

    def generate_path(self, si, qi, dtheta, ds = 3): 
        # (si, qi): 시작상태, dtheta: heading - ryaw, ds: polynomial의 길이, qf: 종료상태 q
        candidate_paths = []
        sl_d = 0.5
        sf_base = si + ds + self.S_MARGIN 

        qf_arr = np.linspace(-3.0, 3.0, 5)
        condition = [np.abs(qf_arr)<=0.1, (np.abs(qf_arr)>0.1)]
        choose = [sf_base+3.0, sf_base-1.0]
        sf_arr = np.select(condition, choose)

        for qf_i, sf_i in zip(qf_arr, sf_arr):
            fp = frenet_path.Frenet_path()
            qs = polynomial.cubic_polynomial(si, qi, dtheta, ds, qf_i)  
            fp.s = np.arange(si, sf_i, sl_d)
            fp.q = qs.calc_point(fp.s) 
            
            fp.x, fp.y = self.glob_path.sl2xy(fp.s, fp.q)
            fp.yaw = self.glob_path.get_current_reference_yaw()
            fp.k = qs.calc_kappa(fp.s, self.glob_path.get_current_reference_kappa())

            fp.offset_cost = abs(qf_i)
            fp.consistency_cost = self.calc_consistency_cost(fp.q, self.last_selected_path.q)
            fp.total_cost = W_CONSISTENCY * fp.consistency_cost + W_OFFSET * fp.offset_cost
            
            candidate_paths.append(fp)

        return candidate_paths
    
    def calc_consistency_cost(self, target_q, last_selected_q):
        consistency_cost = 0
        min_len = min(len(target_q), len(last_selected_q))
        diff = np.abs(target_q[:min_len] - last_selected_q[:min_len])
        
        consistency_cost = np.sum(diff) / len(last_selected_q) if len(last_selected_q) > 0 else 0
        return consistency_cost

    def optimal_trajectory(self, x, y, heading):
        si, qi = self.glob_path.xy2sl(x, y)
        self.current_s = si
        self.current_q = qi
        ryaw = self.glob_path.get_current_reference_yaw_no_s()
        dtheta = heading - ryaw 
        
        selected_path = self.generate_path(si, qi, dtheta, self.ds)
        
        ############### RVIZ 비쥬얼 코드 ##############
        if self.visual == True:
            self.visual_selected_path(selected_path)
        ##############################################

        return selected_path