#!/usr/bin/env python
#-*-coding:utf-8-*-

import numpy as np
from math import tan, sqrt

class quintic_polynomial:

    def __init__(self, xs, vxs, axs, xe, vxe, axe, T):

        # calc coefficient of quintic polynomial
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.xe = xe
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[T**3, T**4, T**5],
                      [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                      [6 * T, 12 * T ** 2, 20 * T ** 3]])
        b = np.array([xe - self.a0 - self.a1 * T - self.a2 * T**2,
                      vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + \
            self.a3 * t**3 + self.a4 * t**4 + self.a5 * t**5

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
            3 * self.a3 * t**2 + 4 * self.a4 * t**3 + 5 * self.a5 * t**4

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2 + 20 * self.a5 * t**3

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t**2

        return xt


class quartic_polynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, T):

        # calc coefficient of quintic polynomial
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * T ** 2, 4 * T ** 3],
                      [6 * T, 12 * T ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + \
            self.a3 * t**3 + self.a4 * t**4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
            3 * self.a3 * t**2 + 4 * self.a4 * t**3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt

class cubic_polynomial:
    def __init__(self, si, qi, dtheta, ds, qf):
        """
        reference: Robot Modeling and Control, Mark W. Spong, Seth Hutchinson, M. Vidyasagar, Wiley, Year: 2005
        Constraints
        q(si) = qi, q'(si) = tan(theta), q(sf) = qf, q'(sf) = 0
        
        Cubic Polynomial
        q(s) = a0 + a1*(s - s0) + a2*(s - s0)**2 + a3*(s - s0)**3
        where
        a2 = (3*(qf - qi) - (2*qi' + qf')*(sf - si)) / (sf - si)**2
        a3 = (2*(qi - qf) + (qi' + qf')*(sf - si)) / (sf - si)**3
        """
        
        self.si, self.ds, self.qf = si, ds, qf
        
        # 경로의 시작을 헤딩과 같은 방향으로 생성하기 위해서
        dtheta = 0

        # calc coefficient of cubic polynomial
        self.a0 = qi
        self.a1 = tan(dtheta)
        self.a2 = (3*(qf - qi) - 2*tan(dtheta)*ds) / ds**2
        self.a3 = (2*(qi - qf) + tan(dtheta)*ds) / ds**3

    def calc_point(self, s):
        s = np.asarray(s)
        delta = s - self.si
        q = (self.a0 + self.a1*delta + self.a2*delta**2 + self.a3*delta**3)
        sf = self.si + self.ds # polynomial이 끝나는 s

        return np.where(s <= sf, q, self.qf)
    
    def calc_kappa(self, s, rk):
        s = np.asarray(s)
        delta = s - self.si
        q = (self.a0 + self.a1*delta + self.a2*delta**2 + self.a3*delta**3)
        q_d = self.a1 + 2*self.a2*delta + 3*self.a3*delta**2
        q_dd = 2*self.a2 + 6*self.a3*delta
        one_minus_qrk = 1 - q*rk
        S = np.sign(one_minus_qrk)
        Q = np.sqrt(q_d**2 + one_minus_qrk**2)

        kappa = S/Q * (rk + (one_minus_qrk*q_dd + rk*(q_d**2)) / Q**2) 
        
        return kappa

class Frenet_path:

    def __init__(self, qf):
        self.s = []
        self.q = []

        self.x = []
        self.y = []
        self.yaw = []
        self.k = []
        
        self.qf = qf
        self.offset_cost = 0
        self.safety_cost = 0
        self.consistency_cost = 0
        self.total_cost = 0

def main():
    '''
    import matplotlib.pyplot as plt
    import os, sys
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    import mathdir.cubic_spline_planner as cubic_spline_planner
    from datadir.global_path import GlobalPath

    glob_path = GlobalPath('/home/macaron/catkin_ws/src/macaron_3/path/manhae2.npy')
 
    frenet_path = []
    si = 5 # 차현재위치
    qi = 10  # offset (l값)

    dtheta = 0
    ds = 10
    sf = si + ds
    qf = 5

    
    for qf_ in range(qf, -(qf + 1), -1):
        fp = Frenet_path()
        qs = cubic_polynomial(si, qi, dtheta, ds, qf_)
        fp.s = [s for s in np.arange(si, sf, 1)]
        fp.q = [qs.calc_point(s) for s in fp.s]
        for i in range(len(fp.s)):
            x, y, yaw, rkappa = glob_path.sl2xy_yaw_kappa(fp.s[i], fp.q[i])
            fp.x.append(x)
            fp.y.append(y)
            fp.yaw.append(yaw)
            fp.k.append(qs.calc_kappa(fp.s[i],rkappa))
        frenet_path.append(fp)
        
    
    for i in range(2*qf + 1):
        plt.plot(frenet_path[i].s, frenet_path[i].q, 'o')
    plt.show()
'''
if __name__ == '__main__':
    main()
                