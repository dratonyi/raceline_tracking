import numpy as np
from numpy.typing import ArrayLike
# from simulator import RaceTrack # Assume imported

# --- Define P-Gains (Tuning Required!) ---
K_P_V = 5.0 
K_P_DELTA = 3.0

# --- Define Lookahead Distance ---
# L is a critical tuning parameter. Start small.
LOOKAHEAD_DISTANCE = 10.0 # meters

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
            
            return lookahead_point

        total_dist += curr_dist
        current_idx = next_idx
        
        if current_idx == closest_idx:
             return centerline[closest_idx] 

    return centerline[closest_idx]


def lower_controller(
    state : ArrayLike, desired : ArrayLike, parameters : ArrayLike
) -> ArrayLike:
    """
    Implements the C1 (Longitudinal) and C2 (Lateral) P-controllers.
    """
    delta = state[2]
    v = state[3]
    delta_r = desired[0]
    v_r = desired[1]

    # C1: Longitudinal Control (P-only)
    a = K_P_V * (v_r - v)
    
    # C2: Lateral Control (P-only)
    v_delta = K_P_DELTA * (delta_r - delta)
    
    return np.array([v_delta, a])


def controller(
    state : ArrayLike, parameters : ArrayLike, racetrack # : RaceTrack
) -> ArrayLike:
    
    # 1. Extract Car State and Path Data
    s_x, s_y, _, v, phi = state # Note: phi is state[4], position is state[0:2]
    current_pos = state[0:2]
    centerline = racetrack.centerline # N x 2 array
    
    lookahead_target = find_lookahead_point(current_pos, centerline, LOOKAHEAD_DISTANCE)

    dx = lookahead_target[0] - s_x
    dy = lookahead_target[1] - s_y
    
    y_target_local = -dx * np.sin(phi) + dy * np.cos(phi)
    l_wb = parameters[0] # Get wheelbase from parameters array

    distance_to_target = np.linalg.norm(lookahead_target - current_pos)
    
    if distance_to_target < 1e-6:
        delta_r = 0.0
    else:
        delta_r = np.arctan(2.0 * l_wb * y_target_local / (distance_to_target**2))
    
    # 4. S1: Calculate Desired Velocity (v_r)
    # This is currently just a placeholder. More advanced logic is needed here.
    v_r = 10.0 
    
    desired_targets = np.array([delta_r, v_r])
    
    # 5. Low-Level Control (C1/C2)
    final_input = lower_controller(state, desired_targets, parameters)
    
    return final_input