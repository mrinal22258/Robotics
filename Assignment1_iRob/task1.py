import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import imageio
import pandas as pd
import os

# ---------------- Controller ----------------
def controller(q, qdot, eta):
    qd = np.pi / 1.5
    kp, kd, ki = 10, 5, 2
    tau = -kp*(q - qd) - kd*qdot - ki*eta
    etadot = q - qd
    return tau, etadot

# ---------------- Dynamics ----------------
def robo_dynamics(q, qdot, tau):
    m, g, l, k = 1, 10, 1, 1
    return (-m*g/l*np.sin(q) - k*qdot + tau) / m

# ---------------- Simulation ----------------
def simulate():
    def ode(t, y):
        q, qdot, eta = y
        tau, etadot = controller(q, qdot, eta)
        qddot = robo_dynamics(q, qdot, tau)
        return [qdot, qddot, etadot]

    y0 = [np.pi/2, 0, 0]        # θ(0)=π/2, θ̇(0)=0
    t = np.linspace(0, 10, 1000)
    sol = solve_ivp(ode, [0, 10], y0, t_eval=t)
    return sol.t, sol.y

# ---------------- Plot Function ----------------
def save_plot(t, y, ylabel, title, filename):
    plt.figure()
    plt.plot(t, y)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ---------------- Animation ----------------
def make_gif(q, filename):
    l = 1
    frames = []
    for theta in q[::10]:
        x1, y1 = l*np.sin(theta), -l*np.cos(theta)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot([-1, 1], [0, 0], 'g', lw=6)
        ax.plot([0, x1], [0, y1], 'b', lw=3)
        ax.plot(x1, y1, 'or', markersize=12)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        fig.canvas.draw()
        frame = np.array(fig.canvas.buffer_rgba())
        frames.append(frame)
        plt.close(fig)
    imageio.mimsave(filename, frames, fps=15)

def main():
    # output folders
    os.makedirs("task1/plots", exist_ok=True)
    os.makedirs("task1/gifs", exist_ok=True)
    os.makedirs("task1/data", exist_ok=True)

    # Run simulation
    t, y = simulate()
    q, qdot = y[0], y[1]

    # Save plots
    save_plot(t, q, "θ (rad)", "Task 1 – θ(t) vs t (Stabilization at π/1.5)", "task1/plots/theta_vs_t.png")
    save_plot(t, qdot, "θ̇ (rad/s)", "Task 1 – θ̇(t) vs t", "task1/plots/theta_dot_vs_t.png")

    # Save animation
    make_gif(q, "task1/gifs/pendulum_stabilization.gif")

    # Save data for analysis
    df = pd.DataFrame({"time": t, "theta": q, "theta_dot": qdot})
    df.to_csv("task1/data/data_stabilization.csv", index=False)


if __name__ == "__main__":
    main()
