import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import imageio
import pandas as pd
import os

# ---------------- Controller ----------------
def controller(q, qdot, eta):
    qd = 0
    kp, kd, ki = 0, -1, 0  # Cancel friction 
    tau = -kp*(q - qd) - kd*qdot - ki*eta
    etadot = q - qd
    return tau, etadot

# ---------------- Dynamics ----------------
def robo_dynamics(q, qdot, tau):
    m, g, l, k = 1, 10, 1, 1
    return (-m*g/l*np.sin(q) - k*qdot + tau) / m

# ---------------- Simulation ----------------
def simulate(theta0):
    def ode(t, y):
        q, qdot, eta = y
        tau, etadot = controller(q, qdot, eta)
        qddot = robo_dynamics(q, qdot, tau)
        return [qdot, qddot, etadot]

    y0 = [theta0, 0, 0]
    t = np.linspace(0, 20, 2000)
    sol = solve_ivp(ode, [0, 20], y0, t_eval=t)
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
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal'); ax.axis('off')
        fig.canvas.draw()
        frame = np.array(fig.canvas.buffer_rgba())
        frames.append(frame)
        plt.close(fig)
    imageio.mimsave(filename, frames, fps=15)

def main():
    # folders
    os.makedirs("task2/plots", exist_ok=True)
    os.makedirs("task2/gifs", exist_ok=True)
    os.makedirs("task2/data", exist_ok=True)

    initials = {
        "pi4": np.pi/4,
        "pi3": np.pi/3
    }
    results = {}

    # Run simulations
    for key, th0 in initials.items():
        t, y = simulate(th0)
        q, qdot = y[0], y[1]
        results[key] = (t, q, qdot)

        # Save θ(t), θ̇(t)
        save_plot(t, q, "θ (rad)", f"Task 2 – θ(t) for θ(0)={round(th0,2)}", f"task2/plots/theta_vs_t_{key}.png")
        save_plot(t, qdot, "θ̇ (rad/s)", f"Task 2 – θ̇(t) for θ(0)={round(th0,2)}", f"task2/plots/theta_dot_vs_t_{key}.png")

        # Save animation
        make_gif(q, f"task2/gifs/pendulum_{key}.gif")

        # Save data
        df = pd.DataFrame({"time": t, "theta": q, "theta_dot": qdot})
        df.to_csv(f"task2/data/data_{key}.csv", index=False)

    # Combined comparison θ(t)
    plt.figure()
    for key, (t, q, qdot) in results.items():
        plt.plot(t, q, label=f"θ(0)={key}")
    plt.title("Task 2 – Combined θ(t) Comparison")
    plt.xlabel("Time (s)"); plt.ylabel("θ (rad)")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig("task2/plots/combined_theta_vs_t.png", dpi=300)
    plt.close()

    # Combined comparison θ̇(t)
    plt.figure()
    for key, (t, q, qdot) in results.items():
        plt.plot(t, qdot, label=f"θ(0)={key}")
    plt.title("Task 2 – Combined θ̇(t) Comparison")
    plt.xlabel("Time (s)"); plt.ylabel("θ̇ (rad/s)")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig("task2/plots/combined_theta_dot_vs_t.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
