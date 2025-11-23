import numpy as np
from numpy.typing import ArrayLike

K_P_V = 10.0 
K_P_DELTA = 3.0
K_D_DELTA = 0.2

LOOKAHEAD_DISTANCE = 10.0
LOOKAHEAD_SPEED = 50.0

DELTA_THRESHOLD = 0.07

STRAIGHT_SPEED = 40.0
TURN_SPEED = 2.0
CURVATURE_GAIN = 60.0

previous_steer_error = 0.0
prev_time_elapsed = 0.0

def find_lookahead_point(current_pos, centerline, L):
    distances_sq = np.sum((centerline - current_pos)**2, axis=1)
    closest_idx = np.argmin(distances_sq)

    N = centerline.shape[0]
    current_idx = closest_idx
    total_dist = 0.0

    while total_dist < L:
        next_idx = (current_idx + 1) % N
        curr_dist = np.linalg.norm(centerline[next_idx] - centerline[current_idx])

        if total_dist + curr_dist >= L:
            remaining_dist = L - total_dist
            ratio = remaining_dist / curr_dist
            lookahead_point = centerline[current_idx] + ratio * (centerline[next_idx] - centerline[current_idx])
            return lookahead_point, closest_idx

        total_dist += curr_dist
        current_idx = next_idx

        if current_idx == closest_idx:
            return centerline[closest_idx], closest_idx

    return centerline[closest_idx], closest_idx

def lower_controller(state: ArrayLike, desired: ArrayLike, parameters: ArrayLike, time_elapsed) -> ArrayLike:
    global previous_steer_error, prev_time_elapsed

    delta = state[2]
    v = state[3]
    delta_r = desired[0]
    v_r = desired[1]

    if abs(delta_r) > DELTA_THRESHOLD:
        a = 0.0
    else:
        a = K_P_V * (v_r - v)

    steer_error = delta_r - delta
    dt = max(time_elapsed - prev_time_elapsed, 1e-6)
    derivative = (steer_error - previous_steer_error) / dt
    v_delta = K_P_DELTA * steer_error + K_D_DELTA * derivative

    previous_steer_error = steer_error
    prev_time_elapsed = time_elapsed

    return np.array([v_delta, a])

def estimate_curvature(p0, p1, p2):
    a = np.linalg.norm(p1 - p0)
    b = np.linalg.norm(p2 - p1)
    c = np.linalg.norm(p2 - p0)
    if a < 1e-6 or b < 1e-6 or c < 1e-6:
        return 0.0
    area = abs(np.cross(p1 - p0, p2 - p0)) / 2.0
    return 4.0 * area / (a * b * c)

def max_abs_curvature_ahead(centerline, start_idx, look_steps=(3, 6, 9, 12, 15)):
    N = len(centerline)
    kappas = []
    for s in look_steps:
        p0 = centerline[start_idx]
        p1 = centerline[(start_idx + s) % N]
        p2 = centerline[(start_idx + 2*s) % N]
        kappas.append(estimate_curvature(p0, p1, p2))
    return max(kappas)

def controller(state: ArrayLike, parameters: ArrayLike, racetrack) -> ArrayLike:
    s_x, s_y, _, v, phi = state
    current_pos = state[0:2]
    centerline = racetrack.centerline
    l_wb = parameters[0]

    lookahead_target, closest_idx = find_lookahead_point(current_pos, centerline, LOOKAHEAD_DISTANCE)
    dx = lookahead_target[0] - s_x
    dy = lookahead_target[1] - s_y
    y_target_local = -dx * np.sin(phi) + dy * np.cos(phi)
    distance_to_target = np.linalg.norm(lookahead_target - current_pos)
    delta_r = 0.0 if distance_to_target < 1e-6 else np.arctan(2.0 * l_wb * y_target_local / (distance_to_target**2))

    speed_lookahead_point, speed_idx = find_lookahead_point(current_pos, centerline, LOOKAHEAD_SPEED)
    kappa_peak = max_abs_curvature_ahead(centerline, speed_idx)
    v_r_raw = STRAIGHT_SPEED / (1.0 + CURVATURE_GAIN * kappa_peak)
    v_r = max(v_r_raw, TURN_SPEED)
    alpha = 0.2
    v_r = alpha * v + (1 - alpha) * v_r

    return np.array([delta_r, v_r])
