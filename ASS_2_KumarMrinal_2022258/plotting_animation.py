import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.animation import FuncAnimation

# ------------------------------------------------------------
# Plot configuration space (obstacles, start, goal, trajectory)
# ------------------------------------------------------------
def plot_configuration_space(obstacles, start, goal, trajectory=None, title="Configuration Space"):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot obstacles
    for obs in obstacles:
        circle = Circle(obs[:2], obs[2], color='red', alpha=0.5)
        ax.add_patch(circle)

    # Start & goal
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax.plot(goal[0], goal[1], 'r*', markersize=12, label='Goal')

    # Trajectory
    if trajectory is not None:
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Path')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim([0, 25])
    ax.set_ylim([0, 25])
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    return fig, ax


# ------------------------------------------------------------
# Plot potential field surface (3D)
# ------------------------------------------------------------
def plot_potential_field(obstacles, goal, ka, kr_array, n0_array, gamma, U_total_func, att_type='parabolic', grid_res=100):

    x = np.linspace(0, 25, grid_res)
    y = np.linspace(0, 25, grid_res)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(grid_res):
        for j in range(grid_res):
            q = np.array([X[i, j], Y[i, j]])
            Z[i, j] = U_total_func(q, goal, obstacles, ka=ka, kr_array=kr_array, n0_array=n0_array, gamma=gamma, att_type=att_type)
    Z = np.clip(Z, 0, 500)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9, edgecolor='none')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Potential')
    ax.set_title(f'Total Potential Field - {att_type}')
    fig.colorbar(surf, ax=ax, shrink=0.5)
    return fig, ax


# ------------------------------------------------------------
# Animate gradient descent motion
# ------------------------------------------------------------
def create_animation(states, trajectory, obstacles, filename='robot_animation.gif', fps=20, dpi=120,
                     max_frames=400, trail_length=100, goal=None):


    if states is None or len(states) == 0:
        print("No robot states provided for animation.")
        return None

    states = np.asarray(states)
    n_states = len(states)

    if goal is None:
        if trajectory is not None and len(trajectory) > 0:
            goal = trajectory[-1]
        else:
            goal = states[-1, :2]

    frames_to_use = min(n_states, max_frames)
    frame_indices = np.linspace(0, n_states - 1, frames_to_use, dtype=int)
    pause_frames = int(fps * 1.5)
    final_frame_indices = np.concatenate([frame_indices, np.full(pause_frames, frame_indices[-1])])

    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)

    for obs in obstacles:
        circ = Circle(obs[:2], obs[2], color='red', alpha=0.3)
        ax.add_patch(circ)

    if trajectory is not None and len(trajectory) > 0:
        ax.plot(trajectory[:, 0], trajectory[:, 1],
                linestyle='--', linewidth=1.5, alpha=0.6, label='Desired Path')

    goal_marker, = ax.plot(goal[0], goal[1], 'r*', markersize=12, label='Goal')

    all_x = np.concatenate([states[:, 0], trajectory[:, 0]]) if trajectory is not None else states[:, 0]
    all_y = np.concatenate([states[:, 1], trajectory[:, 1]]) if trajectory is not None else states[:, 1]

    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)
    margin = 0.15 * max_range
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2

    ax.set_xlim(x_center - max_range/2 - margin, x_center + max_range/2 + margin)
    ax.set_ylim(y_center - max_range/2 - margin, y_center + max_range/2 + margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Bicycle Model Robot Tracking")

    L_body, W_body = 1.2, 0.7
    body_shape = np.array([
        [-L_body/2, -W_body/2],
        [ L_body/2, -W_body/2],
        [ L_body/2,  W_body/2],
        [-L_body/2,  W_body/2],
    ])

    x0, y0, theta0 = states[0]
    R0 = np.array([[np.cos(theta0), -np.sin(theta0)],
                   [np.sin(theta0),  np.cos(theta0)]])
    rotated_body = (R0 @ body_shape.T).T + np.array([x0, y0])
    robot_poly = Polygon(rotated_body, closed=True, color='green', alpha=0.8)
    ax.add_patch(robot_poly)

    heading = ax.quiver(x0, y0, np.cos(theta0), np.sin(theta0),
                        color='black', scale=10, scale_units='xy', width=0.01)

    trail_line, = ax.plot([], [], 'g-', linewidth=2, label='Robot Path')
    ax.legend(loc='upper right')

    trail_x, trail_y = [], []

    def init():
        trail_line.set_data([], [])
        heading.set_offsets(np.array([[x0, y0]]))
        heading.set_UVC(np.cos(theta0), np.sin(theta0))
        robot_poly.set_xy(rotated_body)
        return robot_poly, heading, trail_line, goal_marker

    def animate(frame_idx):
        idx = final_frame_indices[frame_idx]
        x, y, theta = states[idx]

        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        rotated = (R @ body_shape.T).T + np.array([x, y])
        robot_poly.set_xy(rotated)

        trail_x.append(x)
        trail_y.append(y)
        if len(trail_x) > trail_length:
            trail_x.pop(0)
            trail_y.pop(0)
        trail_line.set_data(trail_x, trail_y)

        heading.set_offsets(np.array([[x, y]]))
        heading.set_UVC(np.cos(theta), np.sin(theta))

        return robot_poly, heading, trail_line, goal_marker

    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=len(final_frame_indices),
                         interval=1000.0 / fps, blit=False)

    # ---- Save Animation ----
    try:
        anim.save(filename, writer='pillow', fps=fps)
        print(f"Animation saved: {filename}")
    except Exception as e:
        print(f"Failed to save animation: {e}")

    # Save final image showing desired vs actual path ----
    try:
        final_fig, final_ax = plt.subplots(figsize=(8, 8), dpi=dpi)
        for obs in obstacles:
            circ = Circle(obs[:2], obs[2], color='red', alpha=0.3)
            final_ax.add_patch(circ)
        if trajectory is not None:
            final_ax.plot(trajectory[:, 0], trajectory[:, 1],
                          'b--', label='Desired Path', linewidth=1.8)
        final_ax.plot(states[:, 0], states[:, 1],
                      'g-', label='Actual Path', linewidth=2.5)
        final_ax.plot(goal[0], goal[1], 'r*', markersize=12, label='Goal')
        final_ax.plot(states[0, 0], states[0, 1], 'go', markersize=10, label='Start')

        final_ax.set_xlim(ax.get_xlim())
        final_ax.set_ylim(ax.get_ylim())
        final_ax.set_aspect('equal')
        final_ax.grid(True, alpha=0.3)
        final_ax.set_title("Desired vs Actual Path")
        final_ax.set_xlabel("X")
        final_ax.set_ylabel("Y")
        final_ax.legend(loc='best')

        img_filename = filename.replace('.gif', '_paths.png')
        final_fig.savefig(img_filename, dpi=dpi)
        print(f"saved: {img_filename}")
        plt.close(final_fig)
    except Exception as e:
        print(f"Failed to save image: {e}")

    plt.close(fig)
    return anim
