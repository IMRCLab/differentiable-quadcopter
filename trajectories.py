import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def spline_segment(funcs, coeffs, ts):
    """
    Segment of a spline.

    Parameters:
    -----------
        funcs: array of callables
            list of functions describing a spline
        coeffs: pd.DataFrame
            Pandas DataFrame containing coefficients parameterizing the spline
        ts: np.array
            array containing the descrete time steps the spline should be
            evaluated
    """
    durations = coeffs['duration']
    times = np.cumsum(durations)
    times = np.concat([[0], times])
    num_ts = len(ts)
    t_mask = times.repeat(num_ts).reshape((-1, num_ts))
    active_idxs = (t_mask <= ts).sum(axis=0) - 1
    offsets = times[active_idxs]
    ts_adjusted = ts - offsets

    # construct the coeffs
    coeffs = coeffs.iloc[active_idxs]

    values = [func(coeffs, ts_adjusted) for func in funcs]
    spline = np.stack(values, axis=0).transpose((1,0,2))
    return spline



def f(coeffs, t):
    """
    A seventh order spline parametrized by coefficients.
    """
    x = coeffs['x^0'] + coeffs['x^1'] * t + coeffs['x^2'] * t**2 + coeffs['x^3'] * t**3 + coeffs['x^4'] * t**4 + coeffs['x^5'] * t**5 + coeffs['x^6'] * t**6 + coeffs['x^7'] * t**7
    y = coeffs['y^0'] + coeffs['y^1'] * t + coeffs['y^2'] * t**2 + coeffs['y^3'] * t**3 + coeffs['y^4'] * t**4 + coeffs['y^5'] * t**5 + coeffs['y^6'] * t**6 + coeffs['y^7'] * t**7
    z = coeffs['z^0'] + coeffs['z^1'] * t + coeffs['z^2'] * t**2 + coeffs['z^3'] * t**3 + coeffs['z^4'] * t**4 + coeffs['z^5'] * t**5 + coeffs['z^6'] * t**6 + coeffs['z^7'] * t**7
    yaw = coeffs['yaw^0'] + coeffs['yaw^1'] * t + coeffs['yaw^2'] * t**2 + coeffs['yaw^3'] * t**3 + coeffs['yaw^4'] * t**4 + coeffs['yaw^5'] * t**5 + coeffs['yaw^6'] * t**6 + coeffs['yaw^7'] * t**7
    return np.stack([x,y,z,yaw], axis=1)

def fdot(coeffs, t):
    """
    First derivative (velocity) of a seventh order spline.
    """
    x = coeffs['x^1'] + 2 * coeffs['x^2'] * t + 3 * coeffs['x^3'] * t**2 + 4 * coeffs['x^4'] * t**3 + 5 * coeffs['x^5'] * t**4 + 6 * coeffs['x^6'] * t**5 + 7 * coeffs['x^7'] * t**6
    y = coeffs['y^1'] + 2 * coeffs['y^2'] * t + 3 * coeffs['y^3'] * t**2 + 4 * coeffs['y^4'] * t**3 + 5 * coeffs['y^5'] * t**4 + 6 * coeffs['y^6'] * t**5 + 7 * coeffs['y^7'] * t**6
    z = coeffs['z^1'] + 2 * coeffs['z^2'] * t + 3 * coeffs['z^3'] * t**2 + 4 * coeffs['z^4'] * t**3 + 5 * coeffs['z^5'] * t**4 + 6 * coeffs['z^6'] * t**5 + 7 * coeffs['z^7'] * t**6
    yaw = coeffs['yaw^1'] + 2 * coeffs['yaw^2'] * t + 3 * coeffs['yaw^3'] * t**2 + 4 * coeffs['yaw^4'] * t**3 + 5 * coeffs['yaw^5'] * t**4 + 6 * coeffs['yaw^6'] * t**5 + 7 * coeffs['yaw^7'] * t**6
    return np.stack([x,y,z,yaw], axis=1)

def fdotdot(coeffs, t):
    """
    Second derivative (acceleration) of a seventh order spline.
    """
    x = 2 * coeffs['x^2'] + 6 * coeffs['x^3'] * t + 12 * coeffs['x^4'] * t**2 + 20 * coeffs['x^5'] * t**3 + 30 * coeffs['x^6'] * t**4 + 42 * coeffs['x^7'] * t**5
    y = 2 * coeffs['y^2'] + 6 * coeffs['y^3'] * t + 12 * coeffs['y^4'] * t**2 + 20 * coeffs['y^5'] * t**3 + 30 * coeffs['y^6'] * t**4 + 42 * coeffs['y^7'] * t**5
    z = 2 * coeffs['z^2'] + 6 * coeffs['z^3'] * t + 12 * coeffs['z^4'] * t**2 + 20 * coeffs['z^5'] * t**3 + 30 * coeffs['z^6'] * t**4 + 42 * coeffs['z^7'] * t**5
    yaw = 2 * coeffs['yaw^2'] + 6 * coeffs['yaw^3'] * t + 12 * coeffs['yaw^4'] * t**2 + 20 * coeffs['yaw^5'] * t**3 + 30 * coeffs['yaw^6'] * t**4 + 42 * coeffs['yaw^7'] * t**5
    return np.stack([x,y,z,yaw], axis=1)

def fdotdotdot(coeffs, t):
    """
    Second derivative (jerk) of a seventh order spline.
    """
    x = 6 * coeffs['x^3'] + 24 * coeffs['x^4'] * t + 60 * coeffs['x^5'] * t**2 + 120 * coeffs['x^6'] * t**3 + 210 * coeffs['x^7'] * t**4
    y = 6 * coeffs['y^3'] + 24 * coeffs['y^4'] * t + 60 * coeffs['y^5'] * t**2 + 120 * coeffs['y^6'] * t**3 + 210 * coeffs['y^7'] * t**4
    z = 6 * coeffs['z^3'] + 24 * coeffs['z^4'] * t + 60 * coeffs['z^5'] * t**2 + 120 * coeffs['z^6'] * t**3 + 210 * coeffs['z^7'] * t**4
    yaw = 6 * coeffs['yaw^3'] + 24 * coeffs['yaw^4'] * t + 60 * coeffs['yaw^5'] * t**2 + 120 * coeffs['yaw^6'] * t**3 + 210 * coeffs['yaw^7'] * t**4
    return np.stack([x,y,z,yaw], axis=1)

def plot_trajectory_data(trajectory, ts, title, labels):

    rows = len(labels) 
    fig, axs = plt.subplots(rows,1)
    fig.suptitle(title)
    fig.supxlabel('time [s]')
    for i, label in enumerate(labels):
        axs[i].plot(ts,trajectory[:,i])
        axs[i].set_ylabel(label)
        axs[i].grid()
    
    fig.tight_layout()
    plt.show()


if __name__=="__main__":
    data = pd.read_csv('figure8.csv')
    t_max = np.sum(data['duration'])

    dt = 1/50
    ts = np.arange(0,t_max,dt)

    funcs = [f, fdot, fdotdot, fdotdotdot]
    spline_values = spline_segment(funcs, coeffs=data, ts=ts)
    plot_trajectory_data(spline_values[:,0,:], ts, 'UAV position', labels=['x', 'y','z', 'yaw'])
    plot_trajectory_data(spline_values[:,1,:], ts, 'UAV velocity', labels=['x', 'y','z', 'yaw'])
    plot_trajectory_data(spline_values[:,2,:], ts, 'UAV acceleration', labels=['x', 'y','z', 'yaw'])
    plot_trajectory_data(spline_values[:,3,:], ts, 'UAV jerk', labels=['x', 'y','z', 'yaw'])
