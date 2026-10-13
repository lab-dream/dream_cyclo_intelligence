"""
This file contains the mathematical functions

Author: Suhan Park (park94@kw.ac.kr)
"""
import numpy as np

def quintic_spline(t : float, t_0 : float, t_f : float,
                   x_0 : float,
                   x_dot_0 : float,
                   x_ddot_0 : float,
                   x_f : float,
                   x_dot_f : float,
                   x_ddot_f : float):
    """
    Quintic polynomial spline interpolation.

    Args:
        t (float): Current time.
        t_0 (float): Start time.
        t_f (float): End time.
        x_0 (float): Initial position.
        x_dot_0 (float): Initial velocity.
        x_ddot_0 (float): Initial acceleration.
        x_f (float): Final position.
        x_dot_f (float): Final velocity.
        x_ddot_f (float): Final acceleration.

    Returns:
        np.ndarray: Position, velocity, and acceleration at time t.
    """
    if t < t_0:
        return np.array([x_0, x_dot_0, x_ddot_0])
    
    if t > t_f:
        return np.array([x_f, x_dot_f, x_ddot_f])
    
    t_s = t_f - t_0
    a1 = x_0
    a2 = x_dot_0
    a3 = x_ddot_0 / 2.0
    mat = np.array([
        [t_s**3, t_s**4, t_s**5],
        [3 * t_s**2, 4 * t_s**3, 5 * t_s**4],
        [6 * t_s, 12 * t_s**2, 20 * t_s**3]
    ])
    v = np.array([
        x_f - x_0 - x_dot_0 * t_s - 0.5 * x_ddot_0 * t_s**2,
        x_dot_f - x_dot_0 - x_ddot_0 * t_s,
        x_ddot_f - x_ddot_0
    ])
    res = np.linalg.solve(mat, v)
    a4, a5, a6 = res
    t_e = t - t_0
    position = a1 + a2 * t_e + a3 * t_e**2 + a4 * t_e**3 + a5 * t_e**4 + a6 * t_e**5
    velocity = a2 + 2 * a3 * t_e + 3 * a4 * t_e**2 + 4 * a5 * t_e**3 + 5 * a6 * t_e**4
    acceleration = 2 * a3 + 6 * a4 * t_e + 12 * a5 * t_e**2 + 20 * a6 * t_e**3

    return np.array([position, velocity, acceleration])

def quintic_spline_multi(t, t_0, t_f,
                        x_0 : np.ndarray, 
                        x_dot_0 : np.ndarray, 
                        x_ddot_0 : np.ndarray, 
                        x_f : np.ndarray, 
                        x_dot_f : np.ndarray, 
                        x_ddot_f : np.ndarray):
    if t < t_0:
        return np.array([x_0, x_dot_0, x_ddot_0])
    
    if t > t_f:
        return np.array([x_f, x_dot_f, x_ddot_f])
    
    dim = x_0.shape[0]

    results = np.zeros((3, dim))

    for d in range(dim):
        res = quintic_spline(t, t_0, t_f, x_0[d], x_dot_0[d], x_ddot_0[d], x_f[d], x_dot_f[d], x_ddot_f[d])
        results[:, d] = res

    return results

