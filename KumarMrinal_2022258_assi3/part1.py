import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

# Model Parameters
m1 = 3.473  # kg
m2 = 0.196  # kg
a1 = 1.0    # m
a2 = 1.0    # m
g = 9.81    # m/s^2
f1 = 5.3    # Nm.s
f2 = 1.1    # Nm.s

# PD Controller Gains 
Kp = np.array([150.0, 150.0])  # Proportional gains
Kd = np.array([35.0, 35.0])    # Derivative gains

def inertia_matrix(q):
    """Compute the inertia matrix M(q)"""
    theta2 = q[1]
    M11 = (m1 + m2) * a1**2 + m2 * a2**2 + 2 * m2 * a1 * a2 * np.cos(theta2)
    M12 = m2 * a2**2 + m2 * a1 * a2 * np.cos(theta2)
    M21 = M12
    M22 = m2 * a2**2
    return np.array([[M11, M12], [M21, M22]])

def coriolis_vector(q, q_dot):
    """Compute the Coriolis/centripetal vector V(q, q_dot)"""
    theta2 = q[1]
    theta1_dot = q_dot[0]
    theta2_dot = q_dot[1]
    V1 = -m2 * a1 * a2 * (2 * theta1_dot * theta2_dot + theta2_dot**2) * np.sin(theta2)
    V2 = m2 * a1 * a2 * theta1_dot**2 * np.sin(theta2)
    return np.array([V1, V2])

def friction_matrix(with_friction=True):
    """Compute the friction matrix F"""
    if with_friction:
        return np.array([[f1, 0], [0, f2]])
    else:
        return np.array([[0, 0], [0, 0]])

def gravity_vector(q):
    """Compute the gravity vector G(q)"""
    theta1 = q[0]
    theta2 = q[1]
    G1 = (m1 + m2) * g * a1 * np.cos(theta1) + m2 * g * a2 * np.cos(theta1 + theta2)
    G2 = m2 * g * a2 * np.cos(theta1 + theta2)
    return np.array([G1, G2])

def forward_kinematics(q):
    """Compute end-effector position from joint angles"""
    theta1, theta2 = q[0], q[1]
    x2 = a1 * np.cos(theta1) + a2 * np.cos(theta1 + theta2)
    y2 = a1 * np.sin(theta1) + a2 * np.sin(theta1 + theta2)
    return np.array([x2, y2])

def inverse_kinematics(x, y):
    """Compute joint angles from end-effector position using elbow-down solution"""
    r = np.sqrt(x**2 + y**2)
    
    # Check if target is reachable
    if r > (a1 + a2):
        print(f"Warning: Target ({x:.3f}, {y:.3f}) is outside workspace (r={r:.3f} > {a1+a2})")
        r = a1 + a2 - 0.01
        x = x * r / np.sqrt(x**2 + y**2)
        y = y * r / np.sqrt(x**2 + y**2)
    elif r < abs(a1 - a2):
        print(f"Warning: Target ({x:.3f}, {y:.3f}) is too close (r={r:.3f} < {abs(a1-a2)})")
        r = abs(a1 - a2) + 0.01
        x = x * r / np.sqrt(x**2 + y**2)
        y = y * r / np.sqrt(x**2 + y**2)
    
    # Elbow-down solution
    cos_theta2 = (x**2 + y**2 - a1**2 - a2**2) / (2 * a1 * a2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    theta2 = np.arccos(cos_theta2)
    
    # Compute theta1
    k1 = a1 + a2 * np.cos(theta2)
    k2 = a2 * np.sin(theta2)
    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    
    return np.array([theta1, theta2])

def pd_controller(q, q_dot, q_desired, q_dot_desired):
    """
    PD Controller: tau = -Kp(q - qd) - Kd(q_dot - qd_dot)
    Note: For Part 1, q_dot_desired = 0 (stationary target)
    """
    error = q - q_desired
    error_dot = q_dot - q_dot_desired
    tau = -Kp * error - Kd * error_dot
    return tau

def dynamics(state, t, q_desired, q_dot_desired, with_friction):
    """
    System dynamics: M(q)q_ddot + V(q,q_dot) + F*q_dot + G(q) = tau
    Solving for q_ddot: q_ddot = M^-1 * [tau - V - F*q_dot - G]
    """
    q = state[:2]
    q_dot = state[2:]
    
    # Compute PD control input
    tau = pd_controller(q, q_dot, q_desired, q_dot_desired)
    
    # Compute dynamics matrices
    M = inertia_matrix(q)
    V = coriolis_vector(q, q_dot)
    F = friction_matrix(with_friction)
    G = gravity_vector(q)
    
    # Compute acceleration: M*q_ddot = tau - V - F*q_dot - G
    q_ddot = np.linalg.solve(M, tau - V - F @ q_dot - G)
    
    return np.concatenate([q_dot, q_ddot])

def simulate_point_to_point(initial_q, target_x, target_y, with_friction=True, duration=10.0):
    """
    Part 1: Point-to-point control
    Target: (x2, y2) = (target_x, target_y)
    """
    print(f"\n{'='*70}")
    print(f"Point-to-Point Control (Friction: {with_friction})")
    print(f"{'='*70}")
    
    # Step 1: Apply inverse kinematics to get desired joint angles
    q_desired = inverse_kinematics(target_x, target_y)
    q_dot_desired = np.array([0.0, 0.0])  # Stationary target
    
    print(f"Target position: ({target_x}, {target_y})")
    print(f"Desired joint angles: θ1={np.degrees(q_desired[0]):.2f}°, θ2={np.degrees(q_desired[1]):.2f}°")
    print(f"Initial joint angles: θ1={np.degrees(initial_q[0]):.2f}°, θ2={np.degrees(initial_q[1]):.2f}°")
    
    # Initial state: [q1, q2, q1_dot, q2_dot]
    initial_state = np.concatenate([initial_q, [0.0, 0.0]])
    
    # Time array
    t = np.linspace(0, duration, int(duration * 100))
    
    # Simulate using ODE solver
    states = odeint(dynamics, initial_state, t, 
                    args=(q_desired, q_dot_desired, with_friction))
    
    # Extract results
    q_trajectory = states[:, :2]
    q_dot_trajectory = states[:, 2:]
    
    # Compute end-effector trajectory
    ee_trajectory = np.array([forward_kinematics(q) for q in q_trajectory])
    x2_trajectory = ee_trajectory[:, 0]
    y2_trajectory = ee_trajectory[:, 1]
    
    # Compute control torques
    tau_trajectory = np.array([pd_controller(states[i, :2], states[i, 2:], 
                                              q_desired, q_dot_desired) 
                                for i in range(len(t))])
    
    # Print final results
    print(f"\nFinal position: ({x2_trajectory[-1]:.4f}, {y2_trajectory[-1]:.4f})")
    print(f"Final joint angles: θ1={np.degrees(q_trajectory[-1, 0]):.2f}°, θ2={np.degrees(q_trajectory[-1, 1]):.2f}°")
    error = np.linalg.norm([x2_trajectory[-1] - target_x, y2_trajectory[-1] - target_y])
    print(f"Position error: {error:.6f} m")
    print(f"Max torque: τ1={np.max(np.abs(tau_trajectory[:, 0])):.2f} Nm, τ2={np.max(np.abs(tau_trajectory[:, 1])):.2f} Nm")
    
    return t, x2_trajectory, y2_trajectory, tau_trajectory, q_trajectory

def plot_results(results_with_friction, results_without_friction, target_x, target_y):
    """Plot comparison of results with and without friction"""
    t_w, x2_w, y2_w, tau_w, _ = results_with_friction
    t_wo, x2_wo, y2_wo, tau_wo, _ = results_without_friction
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Part 1: Point-to-Point Control to ({target_x}, {target_y})', 
                 fontsize=16, fontweight='bold')
    
    # x2 vs t
    axes[0, 0].plot(t_w, x2_w, 'b-', linewidth=2, label='With Friction')
    axes[0, 0].plot(t_wo, x2_wo, 'r--', linewidth=2, label='Without Friction')
    axes[0, 0].axhline(y=target_x, color='g', linestyle=':', linewidth=2, label='Target')
    axes[0, 0].set_xlabel('Time (s)', fontsize=12)
    axes[0, 0].set_ylabel('x₂ (m)', fontsize=12)
    axes[0, 0].set_title('End-Effector X Position vs Time', fontsize=13)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # y2 vs t
    axes[0, 1].plot(t_w, y2_w, 'b-', linewidth=2, label='With Friction')
    axes[0, 1].plot(t_wo, y2_wo, 'r--', linewidth=2, label='Without Friction')
    axes[0, 1].axhline(y=target_y, color='g', linestyle=':', linewidth=2, label='Target')
    axes[0, 1].set_xlabel('Time (s)', fontsize=12)
    axes[0, 1].set_ylabel('y₂ (m)', fontsize=12)
    axes[0, 1].set_title('End-Effector Y Position vs Time', fontsize=13)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Torque 1 vs t
    axes[1, 0].plot(t_w, tau_w[:, 0], 'b-', linewidth=2, label='With Friction')
    axes[1, 0].plot(t_wo, tau_wo[:, 0], 'r--', linewidth=2, label='Without Friction')
    axes[1, 0].set_xlabel('Time (s)', fontsize=12)
    axes[1, 0].set_ylabel('τ₁ (Nm)', fontsize=12)
    axes[1, 0].set_title('Control Torque - Joint 1', fontsize=13)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Torque 2 vs t
    axes[1, 1].plot(t_w, tau_w[:, 1], 'b-', linewidth=2, label='With Friction')
    axes[1, 1].plot(t_wo, tau_wo[:, 1], 'r--', linewidth=2, label='Without Friction')
    axes[1, 1].set_xlabel('Time (s)', fontsize=12)
    axes[1, 1].set_ylabel('τ₂ (Nm)', fontsize=12)
    axes[1, 1].set_title('Control Torque - Joint 2', fontsize=13)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('part1_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: part1_results.png")
    plt.show()

def animate_manipulator(q_trajectory, title, filename, with_friction, target_pos=None):
    """Create animation of the manipulator"""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    
    friction_text = "With Friction" if with_friction else "Without Friction"
    ax.set_title(f'{title}\n{friction_text}', fontsize=14, fontweight='bold')
    
    # Plot target if provided
    if target_pos is not None:
        ax.plot(target_pos[0], target_pos[1], 'g*', markersize=20, 
                label='Target', zorder=5)
    
    # Initialize plot elements
    line, = ax.plot([], [], 'o-', linewidth=4, markersize=10, color='blue', label='Manipulator')
    trace, = ax.plot([], [], 'r-', linewidth=1, alpha=0.5, label='Trajectory')
    end_effector, = ax.plot([], [], 'ro', markersize=12)
    
    ax.legend(loc='upper right')
    
    trace_x, trace_y = [], []
    
    def init():
        line.set_data([], [])
        trace.set_data([], [])
        end_effector.set_data([], [])
        return line, trace, end_effector
    
    def animate(frame):
        idx = frame * 5
        if idx >= len(q_trajectory):
            idx = len(q_trajectory) - 1
        
        theta1, theta2 = q_trajectory[idx]
        
        # Compute link positions
        x1 = a1 * np.cos(theta1)
        y1 = a1 * np.sin(theta1)
        x2 = x1 + a2 * np.cos(theta1 + theta2)
        y2 = y1 + a2 * np.sin(theta1 + theta2)
        
        # Update manipulator
        line.set_data([0, x1, x2], [0, y1, y2])
        
        # Update trace
        trace_x.append(x2)
        trace_y.append(y2)
        trace.set_data(trace_x, trace_y)
        
        # Update end-effector
        end_effector.set_data([x2], [y2])
        
        return line, trace, end_effector
    
    frames = min(200, len(q_trajectory) // 5)
    anim = FuncAnimation(fig, animate, init_func=init, frames=frames, 
                        interval=50, blit=True, repeat=True)
    
    anim.save(filename, writer='pillow', fps=20, dpi=100)
    print(f"✓ Saved: {filename}")
    plt.close()

def main():
    """Main function for Part 1"""
    print("="*70)
    print(" PART 1: POINT-TO-POINT PD CONTROL")
    print("="*70)
    
    # Initial configuration (can start from any configuration)
    initial_q = np.array([np.pi/4, np.pi/4])  # 45° for both joints
    
    # Target end-effector position
    target_x, target_y = 1.0, 1.0
    
    # Simulate with friction
    results_with_friction = simulate_point_to_point(
        initial_q, target_x, target_y, with_friction=True, duration=10.0
    )
    
    # Simulate without friction
    results_without_friction = simulate_point_to_point(
        initial_q, target_x, target_y, with_friction=False, duration=10.0
    )
    
    # Plot results
    plot_results(results_with_friction, results_without_friction, target_x, target_y)
    
    # Create animations
    animate_manipulator(results_with_friction[4], 
                       'Part 1: Point-to-Point Control', 
                       'part1_with_friction.gif', 
                       with_friction=True,
                       target_pos=[target_x, target_y])
    
    animate_manipulator(results_without_friction[4], 
                       'Part 1: Point-to-Point Control', 
                       'part1_without_friction.gif', 
                       with_friction=False,
                       target_pos=[target_x, target_y])
    
    print("\n" + "="*70)
    print(" PART 1 COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - part1_results.png")
    print("  - part1_with_friction.gif")
    print("  - part1_without_friction.gif")
    print("="*70)

if __name__ == "__main__":
    main()