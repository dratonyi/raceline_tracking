import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack

K_V = 5.0
K_D = 3.0

def lower_controller(
    state : ArrayLike, desired : ArrayLike, parameters : ArrayLike
) -> ArrayLike:
    # [steer angle, velocity]
    assert(desired.shape == (2,))

    delta = state[2]
    v = state[3]
    delta_r = desired[0]
    v_r = desired[1]

    a = K_V * (v_r - v)

    v_delta = K_D * (delta_r - delta)
    
    return np.array([v_delta, a]).T

def controller(
    state : ArrayLike, parameters : ArrayLike, racetrack : RaceTrack
) -> ArrayLike:
    
    #find lookahead point


    return np.array([0, 100]).T

    def get_lookahead_point(pos, centerline, dist):
        distances_sq = np.sum((centerline - pos)**2, axis=1)
        closest_idx = np.argmin(distances_sq)

        
