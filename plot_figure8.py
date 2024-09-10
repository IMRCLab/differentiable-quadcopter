import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def f(coeffs, t):
    """
    A seventh order spline parametrized by coefficients.
    """
    durations = coeffs['duration']
    times = np.cumsum(durations)
    idx = len(times[times < t])
    coeffs = coeffs.iloc[idx]
    if idx > 0:
        t = t - times[idx-1]
    x = coeffs['x^0'] + coeffs['x^1'] * t + coeffs['x^2'] * t**2 + coeffs['x^3'] * t**3 + coeffs['x^4'] * t**4 + coeffs['x^5'] * t**5 + coeffs['x^6'] * t**6 + coeffs['x^7'] * t**7
    y = coeffs['y^0'] + coeffs['y^1'] * t + coeffs['y^2'] * t**2 + coeffs['y^3'] * t**3 + coeffs['y^4'] * t**4 + coeffs['y^5'] * t**5 + coeffs['y^6'] * t**6 + coeffs['y^7'] * t**7
    z = coeffs['z^0'] + coeffs['z^1'] * t + coeffs['z^2'] * t**2 + coeffs['z^3'] * t**3 + coeffs['z^4'] * t**4 + coeffs['z^5'] * t**5 + coeffs['z^6'] * t**6 + coeffs['z^7'] * t**7
    yaw = coeffs['yaw^0'] + coeffs['yaw^1'] * t + coeffs['yaw^2'] * t**2 + coeffs['yaw^3'] * t**3 + coeffs['yaw^4'] * t**4 + coeffs['yaw^5'] * t**5 + coeffs['yaw^6'] * t**6 + coeffs['yaw^7'] * t**7
    return np.array([[x], [y], [z], [yaw]]) 

def fdot(coeffs, t):
    """
    First derivative (velocity) of a seventh order spline.
    """
    durations = coeffs['duration']
    times = np.cumsum(durations)
    idx = len(times[times < t])
    coeffs = coeffs.iloc[idx]
    if idx > 0:
        t = t - times[idx-1]
    x = coeffs['x^1'] + 2 * coeffs['x^2'] * t + 3 * coeffs['x^3'] * t**2 + 4 * coeffs['x^4'] * t**3 + 5 * coeffs['x^5'] * t**4 + 6 * coeffs['x^6'] * t**5 + 7 * coeffs['x^7'] * t**6
    y = coeffs['y^1'] + 2 * coeffs['y^2'] * t + 3 * coeffs['y^3'] * t**2 + 4 * coeffs['y^4'] * t**3 + 5 * coeffs['y^5'] * t**4 + 6 * coeffs['y^6'] * t**5 + 7 * coeffs['y^7'] * t**6
    z = coeffs['z^1'] + 2 * coeffs['z^2'] * t + 3 * coeffs['z^3'] * t**2 + 4 * coeffs['z^4'] * t**3 + 5 * coeffs['z^5'] * t**4 + 6 * coeffs['z^6'] * t**5 + 7 * coeffs['z^7'] * t**6
    yaw = coeffs['yaw^1'] + 2 * coeffs['yaw^2'] * t + 3 * coeffs['yaw^3'] * t**2 + 4 * coeffs['yaw^4'] * t**3 + 5 * coeffs['yaw^5'] * t**4 + 6 * coeffs['yaw^6'] * t**5 + 7 * coeffs['yaw^7'] * t**6
    return np.array([[x], [y], [z], [yaw]]) 

def fdotdot(coeffs, t):
    """
    Second derivative (acceleration) of a seventh order spline.
    """
    durations = coeffs['duration']
    times = np.cumsum(durations)
    idx = len(times[times < t])
    coeffs = coeffs.iloc[idx]
    # shift time to time on current spline segment
    if idx > 0:
        t = t - times[idx-1]
    x = 2 * coeffs['x^2'] + 6 * coeffs['x^3'] * t + 12 * coeffs['x^4'] * t**2 + 20 * coeffs['x^5'] * t**3 + 30 * coeffs['x^6'] * t**5 + 42 * coeffs['x^7'] * t**5
    y = 2 * coeffs['y^2'] + 6 * coeffs['y^3'] * t + 12 * coeffs['y^4'] * t**2 + 20 * coeffs['y^5'] * t**3 + 30 * coeffs['y^6'] * t**5 + 42 * coeffs['y^7'] * t**5
    z = 2 * coeffs['z^2'] + 6 * coeffs['z^3'] * t + 12 * coeffs['z^4'] * t**2 + 20 * coeffs['z^5'] * t**3 + 30 * coeffs['z^6'] * t**5 + 42 * coeffs['z^7'] * t**5
    yaw = 2 * coeffs['yaw^2'] + 6 * coeffs['yaw^3'] * t + 12 * coeffs['yaw^4'] * t**2 + 20 * coeffs['yaw^5'] * t**3 + 30 * coeffs['yaw^6'] * t**5 + 42 * coeffs['yaw^7'] * t**5
    return np.array([[x], [y], [z], [yaw]]) 

def fdotdotdot(coeffs, t):
    """
    Second derivative (jerk) of a seventh order spline.
    """
    durations = coeffs['duration']
    times = np.cumsum(durations)
    idx = len(times[times < t])
    coeffs = coeffs.iloc[idx]
    # shift time to time on current spline segment
    if idx > 0:
        t = t - times[idx-1]
    x = 6 * coeffs['x^3'] + 24 * coeffs['x^4'] * t + 60 * coeffs['x^5'] * t**2 + 120 * coeffs['x^6'] * t**3 + 210 * coeffs['x^7'] * t**4
    y = 6 * coeffs['y^3'] + 24 * coeffs['y^4'] * t + 60 * coeffs['y^5'] * t**2 + 120 * coeffs['y^6'] * t**3 + 210 * coeffs['y^7'] * t**4
    z = 6 * coeffs['z^3'] + 24 * coeffs['z^4'] * t + 60 * coeffs['z^5'] * t**2 + 120 * coeffs['z^6'] * t**3 + 210 * coeffs['z^7'] * t**4
    yaw = 6 * coeffs['yaw^3'] + 24 * coeffs['yaw^4'] * t + 60 * coeffs['yaw^5'] * t**2 + 120 * coeffs['yaw^6'] * t**3 + 210 * coeffs['yaw^7'] * t**4
    return np.array([[x], [y], [z], [yaw]]) 

data = pd.read_csv('figure8.csv')

x = data['x^0']
y = data['y^0']
plt.plot(x, y)
plt.show()

durations = data['duration']
times = np.cumsum(durations)
tmax = times.max()
ts = np.linspace(0, tmax, 100)
f_t = np.concat([f(data, t) for t in ts], axis=1).T

print(f_t.shape)
x = f_t[:,0]
y = f_t[:,1]

plt.plot(x, y)
plt.show()
