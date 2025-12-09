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

# PD Controller Gains for circular tracking 
Kp = np.array([250.0, 250.0])  # Proportional gains
Kd = np.array([50.0, 50.0])    # Derivative gains

# Circular trajectory parameters
RADIUS = 1.0      # Circle radius (m)
OMEGA = 0.5       # Angular velocity (rad/s) 

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
        r = a1 + a2 - 0.01
        x = x * r / np.sqrt(x**2 + y**2)
        y = y * r / np.sqrt(x**2 + y**2)
    elif r < abs(a1 - a2):
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

def compute_desired_trajectory(t):
    """
    Compute desired circular trajectory at time t
    xd = R*cos(ωt), yd = R*sin(ωt)
    """
    x_desired = RADIUS * np.cos(OMEGA * t)
    y_desired = RADIUS * np.sin(OMEGA * t)
    return x_desired, y_desired

def compute_desired_velocity(t):
    """
    Compute desired velocity in task space
    x_dot = -R*ω*sin(ωt), y_dot = R*ω*cos(ωt)
    """
    x_dot_desired = -RADIUS * OMEGA * np.sin(OMEGA * t)
    y_dot_desired = RADIUS * OMEGA * np.cos(OMEGA * t)
    return x_dot_desired, y_dot_desired

def compute_jacobian(q):
    """
    Compute Jacobian matrix J(q) where [x_dot; y_dot] = J * [theta1_dot; theta2_dot]
    """
    theta1, theta2 = q[0], q[1]
    
    J11 = -a1 * np.sin(theta1) - a2 * np.sin(theta1 + theta2)
    J12 = -a2 * np.sin(theta1 + theta2)
    J21 = a1 * np.cos(theta1) + a2 * np.cos(theta1 + theta2)
    J22 = a2 * np.cos(theta1 + theta2)
    
    return np.array([[J11, J12], [J21, J22]])

def compute_desired_joint_velocity(t, q_desired):
    """
    Compute desired joint velocity from desired task space velocity
    q_dot_desired = J^-1 * [x_dot_desired; y_dot_desired]
    """
    x_dot_des, y_dot_des = compute_desired_velocity(t)
    J = compute_jacobian(q_desired)
    
    # Check if Jacobian is singular
    if abs(np.linalg.det(J)) < 1e-6:
        return np.array([0.0, 0.0])
    
    q_dot_desired = np.linalg.solve(J, np.array([x_dot_des, y_dot_des]))
    return q_dot_desired

def pd_controller(q, q_dot, q_desired, q_dot_desired):
    """
    PD Controller with feedforward and Error Normalization.
    This fixes the 'break' at 180 degrees by handling angle wrapping.
    """
    error = q - q_desired
    
    # Normalize error to range [-pi, pi] ---
    # This prevents the controller from reacting to the 2*pi jump
    # when crossing from +180 to -180 degrees.
    error = (error + np.pi) % (2 * np.pi) - np.pi
    # -----------------------------------------------

    error_dot = q_dot - q_dot_desired
    tau = -Kp * error - Kd * error_dot
    return tau

def dynamics(state, t, with_friction):
    """
    System dynamics with time-varying desired trajectory
    M(q)q_ddot + V(q,q_dot) + F*q_dot + G(q) = tau
    """
    q = state[:2]
    q_dot = state[2:]
    
    # Compute desired trajectory at current time
    x_desired, y_desired = compute_desired_trajectory(t)
    q_desired = inverse_kinematics(x_desired, y_desired)
    q_dot_desired = compute_desired_joint_velocity(t, q_desired)
    
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

def simulate_circular_tracking(initial_q, with_friction=True, duration=15.0):
    """
    Part 2: Circular trajectory tracking
    Circle: center at (0,0), radius = RADIUS
    """
    print(f"\n{'='*70}")
    print(f"Circular Trajectory Tracking (Friction: {with_friction})")
    print(f"{'='*70}")
    print(f"Circle parameters: R={RADIUS} m, ω={OMEGA} rad/s, center=(0,0)")
    print(f"Period: T={2*np.pi/OMEGA:.2f} s")
    print(f"Initial joint angles: θ1={np.degrees(initial_q[0]):.2f}°, θ2={np.degrees(initial_q[1]):.2f}°")
    
    # Initial state: [q1, q2, q1_dot, q2_dot]
    initial_state = np.concatenate([initial_q, [0.0, 0.0]])
    
    # Time array
    t = np.linspace(0, duration, int(duration * 100))
    
    # Simulate using ODE solver
    states = odeint(dynamics, initial_state, t, args=(with_friction,))
    
    # Extract results
    q_trajectory = states[:, :2]
    q_dot_trajectory = states[:, 2:]
    
    # Compute end-effector trajectory
    ee_trajectory = np.array([forward_kinematics(q) for q in q_trajectory])
    x2_trajectory = ee_trajectory[:, 0]
    y2_trajectory = ee_trajectory[:, 1]
    
    # Compute desired trajectory for comparison
    x_desired_traj = RADIUS * np.cos(OMEGA * t)
    y_desired_traj = RADIUS * np.sin(OMEGA * t)
    
    # Compute control torques
    tau_trajectory = []
    for i in range(len(t)):
        x_d, y_d = compute_desired_trajectory(t[i])
        q_d = inverse_kinematics(x_d, y_d)
        q_dot_d = compute_desired_joint_velocity(t[i], q_d)
        tau = pd_controller(states[i, :2], states[i, 2:], q_d, q_dot_d)
        tau_trajectory.append(tau)
    tau_trajectory = np.array(tau_trajectory)
    
    # Compute tracking errors
    tracking_error = np.sqrt((x2_trajectory - x_desired_traj)**2 + 
                             (y2_trajectory - y_desired_traj)**2)
    
    print(f"\nTracking performance:")
    print(f"  Average error: {np.mean(tracking_error):.6f} m")
    print(f"  Max error: {np.max(tracking_error):.6f} m")
    print(f"  RMS error: {np.sqrt(np.mean(tracking_error**2)):.6f} m")
    print(f"  Max torque: τ1={np.max(np.abs(tau_trajectory[:, 0])):.2f} Nm, τ2={np.max(np.abs(tau_trajectory[:, 1])):.2f} Nm")
    
    return (t, x2_trajectory, y2_trajectory, tau_trajectory, q_trajectory, 
            x_desired_traj, y_desired_traj, tracking_error)

def plot_results(results_with_friction, results_without_friction):
    """Plot comparison of results with and without friction"""
    t_w, x2_w, y2_w, tau_w, _, x_des, y_des, err_w = results_with_friction
    t_wo, x2_wo, y2_wo, tau_wo, _, _, _, err_wo = results_without_friction
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Part 2: Circular Trajectory Tracking', fontsize=16, fontweight='bold')
    
    # x2 vs t
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_w, x2_w, 'b-', linewidth=2, label='With Friction')
    ax1.plot(t_wo, x2_wo, 'r--', linewidth=2, label='Without Friction')
    ax1.plot(t_w, x_des, 'g:', linewidth=2, label='Desired')
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('x₂ (m)', fontsize=11)
    ax1.set_title('End-Effector X Position vs Time', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # y2 vs t
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_w, y2_w, 'b-', linewidth=2, label='With Friction')
    ax2.plot(t_wo, y2_wo, 'r--', linewidth=2, label='Without Friction')
    ax2.plot(t_w, y_des, 'g:', linewidth=2, label='Desired')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('y₂ (m)', fontsize=11)
    ax2.set_title('End-Effector Y Position vs Time', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Circular path (X-Y plane)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(x2_w, y2_w, 'b-', linewidth=2, label='With Friction')
    ax3.plot(x2_wo, y2_wo, 'r--', linewidth=2, label='Without Friction')
    ax3.plot(x_des, y_des, 'g:', linewidth=2, label='Desired')
    ax3.plot(x2_w[0], y2_w[0], 'bo', markersize=10, label='Start')
    ax3.set_xlabel('x₂ (m)', fontsize=11)
    ax3.set_ylabel('y₂ (m)', fontsize=11)
    ax3.set_title('End-Effector Path (X-Y Plane)', fontsize=12)
    ax3.axis('equal')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Torque 1 vs t
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t_w, tau_w[:, 0], 'b-', linewidth=2, label='With Friction')
    ax4.plot(t_wo, tau_wo[:, 0], 'r--', linewidth=2, label='Without Friction')
    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_ylabel('τ₁ (Nm)', fontsize=11)
    ax4.set_title('Control Torque - Joint 1', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Torque 2 vs t
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(t_w, tau_w[:, 1], 'b-', linewidth=2, label='With Friction')
    ax5.plot(t_wo, tau_wo[:, 1], 'r--', linewidth=2, label='Without Friction')
    ax5.set_xlabel('Time (s)', fontsize=11)
    ax5.set_ylabel('τ₂ (Nm)', fontsize=11)
    ax5.set_title('Control Torque - Joint 2', fontsize=12)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # Tracking error vs t
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(t_w, err_w, 'b-', linewidth=2, label='With Friction')
    ax6.plot(t_wo, err_wo, 'r--', linewidth=2, label='Without Friction')
    ax6.set_xlabel('Time (s)', fontsize=11)
    ax6.set_ylabel('Tracking Error (m)', fontsize=11)
    ax6.set_title('Tracking Error vs Time', fontsize=12)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # Error statistics (bar plot)
    ax7 = fig.add_subplot(gs[2, :])
    metrics = ['Mean Error', 'Max Error', 'RMS Error']
    with_friction_stats = [np.mean(err_w), np.max(err_w), np.sqrt(np.mean(err_w**2))]
    without_friction_stats = [np.mean(err_wo), np.max(err_wo), np.sqrt(np.mean(err_wo**2))]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    ax7.bar(x_pos - width/2, with_friction_stats, width, label='With Friction', color='blue', alpha=0.7)
    ax7.bar(x_pos + width/2, without_friction_stats, width, label='Without Friction', color='red', alpha=0.7)
    ax7.set_ylabel('Error (m)', fontsize=11)
    ax7.set_title('Tracking Error Statistics', fontsize=12)
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(metrics)
    ax7.legend(fontsize=10)
    ax7.grid(True, axis='y', alpha=0.3)
    
    plt.savefig('part2_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: part2_results.png")
    plt.show()

def animate_manipulator(q_trajectory, title, filename, with_friction, t_array):
    """Create animation of the manipulator following circular trajectory"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Manipulator motion
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    
    friction_text = "With Friction" if with_friction else "Without Friction"
    ax1.set_title(f'{title}\n{friction_text}', fontsize=13, fontweight='bold')
    
    # Plot desired circle
    theta_circle = np.linspace(0, 2*np.pi, 100)
    x_circle = RADIUS * np.cos(theta_circle)
    y_circle = RADIUS * np.sin(theta_circle)
    ax1.plot(x_circle, y_circle, 'g--', linewidth=2, alpha=0.5, label='Desired Circle')
    
    # Initialize manipulator elements
    line, = ax1.plot([], [], 'o-', linewidth=4, markersize=10, color='blue', label='Manipulator')
    trace, = ax1.plot([], [], 'r-', linewidth=1.5, alpha=0.7, label='Actual Path')
    end_effector, = ax1.plot([], [], 'ro', markersize=12)
    
    ax1.legend(loc='upper right', fontsize=10)
    
    # Right plot: Error over time
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Tracking Error (m)', fontsize=12)
    ax2.set_title('Real-time Tracking Error', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    error_line, = ax2.plot([], [], 'b-', linewidth=2)
    
    trace_x, trace_y = [], []
    error_t, error_vals = [], []
    
    # Compute desired trajectory for error calculation
    x_desired_full = RADIUS * np.cos(OMEGA * t_array)
    y_desired_full = RADIUS * np.sin(OMEGA * t_array)
    
    def init():
        line.set_data([], [])
        trace.set_data([], [])
        end_effector.set_data([], [])
        error_line.set_data([], [])
        return line, trace, end_effector, error_line
    
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
        
        # Update error plot
        error = np.sqrt((x2 - x_desired_full[idx])**2 + (y2 - y_desired_full[idx])**2)
        error_t.append(t_array[idx])
        error_vals.append(error)
        error_line.set_data(error_t, error_vals)
        ax2.set_xlim(0, max(error_t[-1], 1))
        ax2.set_ylim(0, max(error_vals) * 1.2 if error_vals else 1)
        
        return line, trace, end_effector, error_line
    
    frames = min(200, len(q_trajectory) // 5)
    anim = FuncAnimation(fig, animate, init_func=init, frames=frames, 
                        interval=50, blit=True, repeat=True)
    
    anim.save(filename, writer='pillow', fps=20, dpi=100)
    print(f"✓ Saved: {filename}")
    plt.close()

def main():
    """Main function for Part 2"""
    print("="*70)
    print(" PART 2: CIRCULAR TRAJECTORY PD CONTROL")
    print("="*70)
    
    # Initial configuration (can start from any configuration)
    initial_q = np.array([np.pi/4, np.pi/4])  # 45° for both joints
    
    # Simulate with friction
    results_with_friction = simulate_circular_tracking(
        initial_q, with_friction=True, duration=15.0
    )
    
    # Simulate without friction
    results_without_friction = simulate_circular_tracking(
        initial_q, with_friction=False, duration=15.0
    )
    
    # Plot results
    plot_results(results_with_friction, results_without_friction)
    
    # Create animations
    animate_manipulator(results_with_friction[4], 
                       'Part 2: Circular Trajectory Tracking', 
                       'part2_with_friction.gif', 
                       with_friction=True,
                       t_array=results_with_friction[0])
    
    animate_manipulator(results_without_friction[4], 
                       'Part 2: Circular Trajectory Tracking', 
                       'part2_without_friction.gif', 
                       with_friction=False,
                       t_array=results_without_friction[0])
    
    print("\n" + "="*70)
    print(" PART 2 COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - part2_results.png")
    print("  - part2_with_friction.gif")
    print("  - part2_without_friction.gif")
    print("="*70)

if __name__ == "__main__":
    main()