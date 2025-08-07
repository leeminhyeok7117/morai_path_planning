#!/usr/bin/env python
#-*-coding:utf-8-*-
"""
Created on Mon Aug 24 15:12:36 2020

@author IHS
"""

class Frenet_path:
    
    def __init__(self):
        self.s = []
        self.splus = []
        self.q = []
        self.qplus = []
        
        self.x = []
        self.xplus = []
        self.y = []
        self.yplus = []
        self.yaw = []
        self.yawplus = []
        self.k = []
        self.kplus = []
        
        self.offset_cost = 0
        self.obs_distance = 0
        self.consistency_cost = 0
        self.total_cost = 0
        self.obstacle_cost = 0
        self.middle_obstacle_cost = 0
        self.lane_judge_cost = 0
        self.direction_cost = 0