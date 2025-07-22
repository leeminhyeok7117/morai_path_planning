import numpy as np
import matplotlib.pyplot as plt
from optimize_with_do_mpc import PathpOptimizer_MPC  # 클래스가 정의된 파일
from cubic_hermit_planner import hermite_with_constraints

def test_mpc_single_goal():
    # 1. 목표 설정
    start = [0, 0]
    goal = [10.0, 5.0]
    start_yaw = 0.0
    goal_yaw = np.deg2rad(20.0)

    # 2. Hermite spline 기반 reference 경로 생성
    x_ref, y_ref, yaw_ref, _ = hermite_with_constraints(start, goal, start_yaw, goal_yaw)

    # 3. MPC 모델 및 컨트롤러 구성
    optimizer = PathpOptimizer_MPC()
    model = optimizer.setup_model()
    mpc = optimizer.configure_mpc(model, x_ref, y_ref, yaw_ref)

    # 4. 초기 상태 설정
    dt = 0.1
    x0 = np.array([0.0, 0.0, 0.0, 5.0])  # x, y, yaw, v
    mpc.x0 = x0
    mpc.set_initial_guess()

    # 5. MPC 수행 및 경로 저장
    mpc_x, mpc_y = [x0[0]], [x0[1]]

    for i in range(len(x_ref) - 1):
        try:
            u = mpc.make_step(x0)

            if u is None or np.any(np.isnan(u)) or np.any(np.isinf(u)):
                print(f"❌ Invalid control input at step {i}: {u}")
                break

            a = u[0].item()
            steer = u[1].item()

            x, y, yaw, v = x0

            # Euler integration
            x += v * np.cos(yaw) * dt
            y += v * np.sin(yaw) * dt
            yaw += v * np.tan(steer) / optimizer.L * dt
            v += a * dt

            x0 = np.array([x, y, yaw, v])
            print(f"✅ Step {i}: x0 = {x0}")

            mpc_x.append(x)
            mpc_y.append(y)

        except Exception as e:
            print(f"❌ Exception at step {i}: {e}")
            break
    # 6. 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(x_ref, y_ref, '--k', label='Hermite Reference Path')
    plt.plot(mpc_x, mpc_y, '-r', label='MPC Optimized Path')
    plt.scatter(start[0], start[1], c='blue', label='Start')
    plt.scatter(goal[0], goal[1], c='green', marker='x', label='Goal')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis('equal')
    plt.grid(True)
    plt.title('Hermite Reference Path vs MPC Optimized Path')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    test_mpc_single_goal()
