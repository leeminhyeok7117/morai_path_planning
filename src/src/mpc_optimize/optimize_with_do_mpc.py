import do_mpc
import numpy as np
from casadi import cos, sin, tan, sumsqr
from cubic_hermit_planner import hermite_with_constraints

class PathpOptimizer_MPC:
    def __init__(self):
        #차량의 동역학적 제약
        self.L = 1.04
        self.max_steer_angle = np.deg2rad(27)
        self.max_speed = 20 / 3.6
        self.max_minus_accel = -(20 / 3.6)
        self.max_plus_accel = 5 / 3.6
        self.max_curvature = np.tan(self.max_steer_angle) / self.L
                
    def setup_model(self):
        model = do_mpc.model.Model("continuous")
        
        #상태변수 정의: 위치(x,y), 속도(v), 헤딩값(heading)  
        self.x = model.set_variable(var_type='_x', var_name='x')
        self.y = model.set_variable(var_type='_x', var_name='y')
        self.yaw = model.set_variable(var_type='_x', var_name='yaw')
        self.v = model.set_variable(var_type='_x', var_name='v')
          
        #제어변수 정의: 가속도(a), 조향각(yaw)
        self.a = model.set_variable(var_type='_u', var_name='a')
        self.steer = model.set_variable(var_type='_u', var_name='steer')
        
        #ordinary diffrential Equation
        model.set_rhs('x', self.v * cos(self.yaw))
        model.set_rhs('y', self.v * sin(self.yaw))
        model.set_rhs('yaw', self.v * tan(self.steer) / self.L)
        model.set_rhs('v', self.a)

        model.setup()
        return model

    def configure_mpc(self, model, x_ref, y_ref, yaw_ref):
        mpc = do_mpc.controller.MPC(model)
        
        setup_mpc = {
            "n_horizon": len(x_ref), 
            "t_step": 0.1,
            "state_discretization": "collocation",  
            "collocation_type": "radau",              
            "collocation_deg": 3,                     
            "store_full_solution": True,
            "nlpsol_opts": {
                "ipopt.print_level": 0,  
                "ipopt.sb": "yes",        
                "print_time": 0,         
            }
        }
        mpc.set_param(**setup_mpc)
        
        mterm = (self.x - x_ref[-1])**2 + (self.y - y_ref[-1])**2 + (self.yaw - yaw_ref[-1])**2 + 0.1 * (self.v - self.max_speed)**2
        lterm = sumsqr(self.x - x_ref[0]) + sumsqr(self.y - y_ref[0]) + sumsqr(self.yaw - yaw_ref[0] + 0.1 * (self.v - self.max_speed)**2)
       
        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.set_rterm(a=1e-4, steer=1e-4)
        
        #최대, 최소 범위 제한
        mpc.bounds['lower', '_u', 'a'] = self.max_minus_accel     # 최대 감속
        mpc.bounds['upper', '_u', 'a'] = self.max_plus_accel     # 최대 가속
        
        mpc.bounds['lower', '_x', 'v'] = 0             # 최소 속도
        mpc.bounds['upper', '_x', 'v'] = self.max_speed     # 최대 속도

        mpc.bounds['lower', '_u', 'steer'] = -self.max_steer_angle  # 최소 조향각
        mpc.bounds['upper', '_u', 'steer'] = self.max_steer_angle  # 최대 조향각  
        
        curvature_expr = tan(self.steer) / self.L
        mpc.set_nl_cons('curvature_limit_upper', curvature_expr - self.max_curvature)
        mpc.set_nl_cons('curvature_limit_lower', -self.max_curvature - curvature_expr)
        
        mpc.setup()
        
        return mpc

    def MPC(self, state):
        x_ref, y_ref, yaw_ref, _ = hermite_with_constraints([0,0], [state[0],state[1]], 0, state[2])
        
        erp_model = self.setup_model()
        erp_controller = self.configure_mpc(erp_model, x_ref, y_ref, yaw_ref)
        
        initial_state = np.array([0.0, 0.0, 0.0, 0.0])  # 초기 위치 x=0, y=0, 속도 v=0, 조향각 yaw=0
        erp_controller.x0 = initial_state
        
        erp_controller.set_initial_guess()
        
        trajectory_x = [] ###
        trajectory_y = [] ###

        for _ in range(100):
            optimal_control = erp_controller.make_step(initial_state)
            
            dt = 0.1  # time step (same as in MPC config)
            x = initial_state[0]
            y = initial_state[1]
            yaw = initial_state[2]
            v = initial_state[3]
            a = optimal_control[0].item()
            steer = optimal_control[1].item()

            # dynamics
            x += v * np.cos(yaw) * dt
            y += v * np.sin(yaw) * dt
            yaw += v * np.tan(steer) / self.L * dt
            v += a * dt
            
            trajectory_x.append(x) ###
            trajectory_y.append(y) ###

            initial_state = np.array([x, y, yaw, v])

            print(f"Step {_}: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}, v={v:.2f}, a={a:.2f}, steer={steer:.2f}")
        
        return trajectory_x, trajectory_y, x_ref, y_ref #optimal_control

