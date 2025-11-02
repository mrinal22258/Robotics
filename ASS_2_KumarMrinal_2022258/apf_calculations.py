import numpy as np

START = np.array([1.0, 1.0])      # q(0)
GOAL  = np.array([20.0, 20.0])    # qg

# Obstacles: [center_x, center_y, radius]
# Each obstacle i corresponds to a convex set Coi (circle here).
OBSTACLES = np.array([
    [4.5, 3.0, 2.0],
    [3.0, 12.0, 2.0],
    [15.0, 15.0, 3.0]
])


# Attractive gain (ka)
KA = 1.0

# For conical attractive Ua = ka * ||q - qg||
# kr_i: repulsive gain for obstacle i (same length as OBSTACLES)
KR = np.array([100.0, 100.0, 150.0])  # example: stronger for the bigger obstacle

# n0_i: influence radius for each obstacle i
N0 = np.array([3.0, 3.0, 4.0]) 

# gamma (shape exponent for repulsive potential)
GAMMA = 4.0 

# Integration / flow settings
DT = 0.02          # integration time-step for explicit Euler (continuous-time approx.)
MAX_STEPS = 10000
GOAL_THRESHOLD = 0.3


# Notation

# q    : current coordinate (2D numpy array)
# qg   : goal coordinate (GOAL)
# Ua   : attractive potential
# Ur   : repulsive potential (sum over obstacles)
# ni(q): distance of q from the convex set Coi (for circles: ||q - c|| - r)
# n0_i : influence distance (scalar for obstacle i)
# kr_i : repulsive gain for obstacle i
# gamma: exponent in repulsive potential
# ka   : attractive gain


# Parabolic attractive:
#   Ua = 1/2 * ka * ||q - qg||^2
#   grad_q Ua = ka * (q - qg)
#   => contribution to qdot (gradient-flow, u(t) = -grad Ua) : ka * (qg - q)
#
# Conical attractive:
#   Ua = ka * ||q - qg||
#   grad_q Ua = ka * (q - qg) / ||q - qg||
#   => contribution to qdot: ka * (qg - q) / ||q - qg||
#
# Repulsive (per obstacle i):
#   Ur,i = (kr_i / gamma) * ( 1/ni(q) - 1/n0_i )^gamma    if ni(q) <= n0_i
#          = 0                                            if ni(q) > n0_i
#
#   Let f = (1/ni - 1/n0)^gamma
#   dUr,i/dq = (kr_i / gamma) * gamma * (1/ni - 1/n0)^(gamma-1) * d(1/ni)/dq
#            = kr_i * (1/ni - 1/n0)^(gamma-1) * (-1/ni^2) * d ni/dq
#            = - kr_i * (1/ni - 1/n0)^(gamma-1) * (1/ni^2) * d ni/dq
#
# For circular Coi (center c, radius r):
#   ni(q) = ||q - c|| - r
#   d ni/dq = (q - c) / ||q - c||
#
# Total potential U = Ua + sum_i Ur,i
# Total gradient: grad U = grad Ua + sum_i grad Ur,i
# Gradient-flow ODE (we solve numerically):
#   qdot = u(t) = - grad U
# ------------------------------

# ------------------------------
# Attractive potentials & gradients
# ------------------------------
def Ua_parabolic(q, qg, ka=KA):
    """Ua = 1/2 * ka * ||q - qg||^2"""
    return 0.5 * ka * np.linalg.norm(q - qg)**2

def grad_Ua_parabolic(q, qg, ka=KA):
    """grad_q Ua = ka * (q - qg)"""
    return ka * (q - qg)

def Ua_conical(q, qg, ka=KA):
    """Ua = ka * ||q - qg||"""
    return ka * np.linalg.norm(q - qg)

def grad_Ua_conical(q, qg, ka=KA, eps=1e-9):
    """grad_q Ua = ka * (q - qg) / ||q - qg||  (handle q==qg)"""
    diff = q - qg
    dist = np.linalg.norm(diff)
    if dist < eps:
        return np.zeros_like(q)   # gradient undefined at q==qg; return zero
    return ka * (diff / dist)

# ------------------------------
# Repulsive (per-obstacle) and its gradient
# ------------------------------
def ni_circle(q, center, radius, eps=1e-9):
    """
    ni(q) for circular Coi:
        ni(q) = ||q - c|| - r
    returns (ni, dist) where dist = ||q - c|| used for derivative
    """
    diff = q - center
    dist = np.linalg.norm(diff)
    ni = dist - radius
    return ni, dist

def Ur_i_and_grad(q, center, radius, kr_i, n0_i, gamma=GAMMA, eps=1e-9):
    """
    Ur,i and grad_q Ur,i for a circular obstacle (Coi).
    
    """
    diff = q - center
    dist = np.linalg.norm(diff)
    
    # Handle being at center
    if dist < eps:
        # Push away in arbitrary direction
        d_nidq = np.array([1.0, 0.0])
        ni = -radius
    else:
        d_nidq = diff / dist  # Unit vector pointing AWAY from center
        ni = dist - radius
    
    # Outside influence zone -> no effect
    if ni > n0_i:
        return 0.0, np.zeros(2)
    
    # Inside or very close to obstacle - use strong repulsion
    if ni <= 0:
        # small positive value to avoid division by zero
        # repulsion very strong when inside
        ni_safe = max(eps, ni + radius * 0.01)  # Small positive value
        term = (1.0 / ni_safe - 1.0 / n0_i)
        Ur_i = (kr_i / gamma) * (term ** gamma)
        
        # Gradient: ∇Ur,i = kr_i * term^(γ-1) * (-1/ni²) * ∇ni
        # Note: This gradient points TOWARD obstacle, but gradient flow (qdot = -∇U) 
        # will push robot AWAY
        grad_Ur_i = kr_i * (term ** (gamma - 1.0)) * (-1.0 / (ni_safe ** 2)) * d_nidq
        
        return Ur_i, grad_Ur_i
    
    # Normal case: outside obstacle but within influence zone
    term = (1.0 / ni - 1.0 / n0_i)
    
    if term < eps:  # Avoid numerical issues
        return 0.0, np.zeros(2)
    
    Ur_i = (kr_i / gamma) * (term ** gamma)
    
    # ∇Ur,i = (kr_i/γ) * γ * term^(γ-1) * ∂(1/ni)/∂q
    #       = kr_i * term^(γ-1) * (-1/ni²) * ∂ni/∂q
    #       = kr_i * term^(γ-1) * (-1/ni²) * d_nidq
    # 
    # This gradient points TOWARD the obstacle.
    # In gradient flow qdot = -∇U, the negative sign makes robot move AWAY.
    grad_Ur_i = kr_i * (term ** (gamma - 1.0)) * (-1.0 / (ni ** 2)) * d_nidq

    return Ur_i, grad_Ur_i



def Ur_total_and_grad(q, obstacles, kr_array, n0_array, gamma=GAMMA):
    """
    Sum Ur,i and grad Ur,i over all obstacles:
      Ur(q) = sum_i Ur,i
      grad Ur(q) = sum_i grad Ur,i
    """
    total_Ur = 0.0
    total_grad = np.zeros(2)
    for idx, obs in enumerate(obstacles):
        center = obs[:2]
        radius = obs[2]
        kr_i = kr_array[idx]
        n0_i = n0_array[idx]
        Ur_i, grad_Ur_i = Ur_i_and_grad(q, center, radius, kr_i, n0_i, gamma=gamma)
        total_Ur += Ur_i
        total_grad += grad_Ur_i
    return total_Ur, total_grad


# ------------------------------
# Total potential and total gradient
# ------------------------------
def U_total(q, qg, obstacles, ka=KA, kr_array=KR, n0_array=N0, gamma=GAMMA, att_type='parabolic'):
    """Return Ua + Ur"""
    if att_type == 'parabolic':
        Ua = Ua_parabolic(q, qg, ka)
    elif att_type == 'conical':
        Ua = Ua_conical(q, qg, ka)
    else:
        raise ValueError("att_type must be 'parabolic' or 'conical'")
    Ur, _ = Ur_total_and_grad(q, obstacles, kr_array, n0_array, gamma=gamma)
    return Ua + Ur

def grad_U_total(q, qg, obstacles, ka=KA, kr_array=KR, n0_array=N0, gamma=GAMMA, att_type='parabolic'):
    """
    Compute gradient of total potential:
      grad U = grad Ua + sum_i grad Ur,i
    Note: grad Ua (parabolic) = ka * (q - qg)
          grad Ua (conical)   = ka * (q - qg) / ||q - qg||
    """
    if att_type == 'parabolic':
        grad_Ua = grad_Ua_parabolic(q, qg, ka)
    elif att_type == 'conical':
        grad_Ua = grad_Ua_conical(q, qg, ka)
    else:
        raise ValueError("att_type must be 'parabolic' or 'conical'")

    _, grad_Ur = Ur_total_and_grad(q, obstacles, kr_array, n0_array, gamma=gamma)
    grad_total = grad_Ua + grad_Ur
    return grad_total

# ------------------------------
# Gradient-flow integrator (continuous-time)
# ------------------------------
def generate_trajectory_gradient_flow(start_q, qg, obstacles,
                                      ka=KA, kr_array=KR, n0_array=N0, gamma=GAMMA,
                                      att_type='conical', dt=DT, max_steps=MAX_STEPS,
                                      goal_threshold=GOAL_THRESHOLD):
    """
    Solve the ODE (gradient-flow):
      qdot = u(t) = - grad_q (Ua + Ur)
    using explicit Euler (small dt) to produce a continuous-time trajectory.
    
    Added collision avoidance and stuck detection
    """
    q = start_q.astype(float).copy()
    trajectory = [q.copy()]

    for step in range(max_steps):
        # check goal
        if np.linalg.norm(q - qg) < goal_threshold:
            break

        # compute total gradient (grad Ua + grad Ur)
        gradU = grad_U_total(q, qg, obstacles, ka=ka, kr_array=kr_array, 
                            n0_array=n0_array, gamma=gamma, att_type=att_type)

        # gradient-flow: qdot = - gradU
        qdot = -gradU
        
        # Limit velocity for numerical stability
        speed = np.linalg.norm(qdot)
        if speed > 5.0:  # Maximum speed limit
            qdot = (qdot / speed) * 5.0

        # integrate (explicit Euler)
        q_new = q + dt * qdot
        
        # Check if new position would be inside obstacle
        collision = False
        for idx, obs in enumerate(obstacles):
            c = obs[:2]
            r = obs[2]
            if np.linalg.norm(q_new - c) < r * 1.05:  # 5% safety margin
                collision = True
                break
        
        if collision:
            # Don't move into obstacle - try tangential movement instead
            # Find direction tangent to obstacle
            closest_obs_idx = np.argmin([np.linalg.norm(q - obs[:2]) for obs in obstacles])
            c = obstacles[closest_obs_idx][:2]
            to_obs = c - q
            # Tangent perpendicular to radial direction
            tangent = np.array([-to_obs[1], to_obs[0]])
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm > 1e-9:
                tangent = tangent / tangent_norm
                q_new = q + dt * tangent * 2.0
            else:
                q_new = q  # Stay in place
        
        q = q_new

        # Ensure q is not inside obstacle (safety enforcement)
        for idx, obs in enumerate(obstacles):
            c = obs[:2]
            r = obs[2]
            diff = q - c
            dist = np.linalg.norm(diff)
            if dist < r:
                # Push to boundary with safety margin
                if dist < 1e-6:
                    diff = np.array([1e-3, 0.0])
                    dist = 1e-6
                q = c + (diff / dist) * (r + 0.05)

        trajectory.append(q.copy())
        
        # Detect if stuck (not making progress)
        if step > 50 and step % 25 == 0:
            recent_dist = np.linalg.norm(trajectory[-1] - trajectory[-25])
            if recent_dist < 0.05:
                print(f"Warning: Possibly stuck at step {step}, adding perturbation")
                # small random perturbation to escape local minimum
                q += np.random.randn(2) * 0.3

    return np.array(trajectory)


def simulate_bicycle_following(traj, v_ref=1.0, L=1.0, dt=0.1, max_time=200.0,
                               goal_threshold=0.3, kx=1.0, ky=3.0, ktheta=2.0):
    """
    Simulates a bicycle model following a reference trajectory using
    a proportional controller that respects non-holonomic constraints.

    Args:
      traj : Nx2 array of (x, y) waypoints from APF (path to follow).
      v_ref : nominal forward velocity.
      L : wheelbase of the vehicle.
      dt : timestep for simulation.
      max_time : simulation stop time.
      goal_threshold : distance to stop when near goal.
      kx, ky, ktheta : proportional controller gains.

    Returns:
      states : Mx3 array of [x, y, theta] over time.
    """

    traj = np.asarray(traj)
    if traj.ndim != 2 or traj.shape[1] != 2:
        raise ValueError("traj must be an Nx2 array")

    # Compute path headings and reference angular velocities
    diff = np.gradient(traj, axis=0)
    theta_ref = np.arctan2(diff[:, 1], diff[:, 0])
    # smooth heading to avoid jumps
    theta_ref = np.unwrap(theta_ref)

    # approximate reference angular velocity ω_ref = dθ/dt for path curvature
    omega_ref = np.gradient(theta_ref) / np.maximum(1e-6, np.hypot(diff[:, 0], diff[:, 1]))

    # initial state
    x, y = traj[0, 0], traj[0, 1]
    theta = theta_ref[0]
    states = [[x, y, theta]]

    total_steps = int(max_time / dt)

    for step in range(total_steps):
        # find closest reference point
        dists = np.hypot(traj[:, 0] - x, traj[:, 1] - y)
        idx = np.argmin(dists)

        if dists[idx] < goal_threshold and idx >= len(traj) - 2:
            break  # goal reached

        # reference state at that index
        xd, yd = traj[idx]
        thetad = theta_ref[idx]
        vd = v_ref
        wd = omega_ref[idx] * v_ref  # desired angular velocity (approx)

        # compute pose error in robot's local frame
        dx = xd - x
        dy = yd - y
        x_e = np.cos(theta) * dx + np.sin(theta) * dy
        y_e = -np.sin(theta) * dx + np.cos(theta) * dy
        theta_e = thetad - theta
        theta_e = np.arctan2(np.sin(theta_e), np.cos(theta_e))  # normalize

        # control law (Samson / Kanayama form)
        v = vd * np.cos(theta_e) + kx * x_e
        omega = wd + vd * (ky * y_e + ktheta * np.sin(theta_e))

        omega = np.clip(omega, -2.0, 2.0)
        v = max(0.0, v)  # prevent reversing unless needed

        # update kinematic model
        x = x + v * np.cos(theta) * dt
        y = y + v * np.sin(theta) * dt
        theta = theta + (v / L) * np.tan(np.arctan(L * omega / max(v, 1e-3))) * dt
        theta = np.arctan2(np.sin(theta), np.cos(theta))

        states.append([x, y, theta])

    return np.array(states)
