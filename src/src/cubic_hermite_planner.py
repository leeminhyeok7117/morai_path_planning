import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline

def hermite_with_constraints(start, end, yaw_start, yaw_end):
    """
    Generate a cubic Hermite spline with yaw and curvature constraints.

    Parameters:
        start: 시작점 좌표
        end: 끝점 좌표
        yaw_start: 시작점 yaw값
        yaw_end: 끝점 yaw값
        max_yaw: 가능한 최대 yaw값 (ERP는 25도)
        wheelbase: 휠베이스
        num_points: 보간되는 점의 총 개수

    Returns:    
        x_vals: 보간된 x값
        y_vals: 보간된 y값
        yaw_vals: 보간된 각 점의 yaw값
        curvature_vals: 보간된 각 점의 curvature값
    """
    # 파라미터 값
    wheelbase = 1.04
    max_yaw = np.deg2rad(25)
    num_points = 25

    # 시작과 끝점의 좌표, yaw값의 tangent값
    x = [start[0], end[0]]
    y = [start[1], end[1]]
    inc = [np.tan(yaw_start), np.tan(yaw_end)]  
    
    # scipy에 내장된 cubic_hermit_spline 사용
    spline = CubicHermiteSpline(x, y, inc)
    
    # 보간된 x,y값 (x값을 기준으로 보간을 진행)
    x_vals = np.linspace(x[0], x[-1], num_points)
    y_vals = spline(x_vals)
    
    # 보간된 각 점마다의 yaw, curvature값 구하기
    dydx_vals = spline.derivative(1)(x_vals)
    yaw_vals = np.arctan(dydx_vals)

    ddyddx_vals = spline.derivative(2)(x_vals)
    curvature_vals = np.abs(ddyddx_vals) / ((1 + dydx_vals**2)**(3/2))
    
    # yaw, curvature값 limit제한, 
    yaw_vals = np.clip(yaw_vals, -max_yaw, max_yaw)
    
    max_curvature = np.tan(max_yaw) / wheelbase   #bicycle model의 곡률 (tan(yaw)/wheelbase)
    curvature_vals = np.clip(curvature_vals, None, max_curvature)
    
    return x_vals, y_vals, yaw_vals, curvature_vals





