import argparse

import matplotlib.pyplot as plt
import numpy as np

import cfusdlog

# pwm normalized [0-1]; vbat normalized [0-1]
# output in N
def pwm2force(pwm, vbat):
    C_00 = 11.093358483549203
    C_10 = -39.08104165843915
    C_01 = -9.525647087583181
    C_20 = 20.573302305476638
    C_11 = 38.42885066644033
    return (C_00 + C_10*pwm + C_01*vbat + C_20*pwm**2 + C_11*vbat*pwm) / 1000 * 9.81

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("file_usd")
    args = parser.parse_args()

    data_usd = cfusdlog.decode(args.file_usd)

    start_time = np.inf
    for _,v in data_usd.items():
        start_time = min(start_time, v['timestamp'][0])

    time_fF = (data_usd['fixedFrequency']['timestamp'] - start_time) / 1e3
    time_eG = (data_usd['estGyroscope']['timestamp'] - start_time) / 1e3

    T = len(data_usd['fixedFrequency']['timestamp'])
    k2 = 0
    gyro = []
    for k1 in range(T-1):
        t_tF = data_usd['fixedFrequency']['timestamp'][k1+1]
        gyro_data = []
        while True:
            t_eG = data_usd['estGyroscope']['timestamp'][k2]
            gyro_data.append(data_usd['estGyroscope']['gyro.x'][k2])
            if t_eG > t_tF:
                break
            k2 = k2 + 1
        print(gyro_data)
        gyro.append(np.mean(gyro_data))




    fig, ax = plt.subplots(3, 1)
    ax[0].plot(time_fF, np.degrees(data_usd['fixedFrequency']['stateEstimateZ.rateRoll'] / 1000))
    ax[0].plot(time_eG, data_usd['estGyroscope']['gyro.x'])
    ax[0].plot(time_fF[1:], gyro)

    ax[1].plot(time_fF, np.degrees(data_usd['fixedFrequency']['stateEstimateZ.ratePitch'] / 1000))
    ax[1].plot(time_eG, -data_usd['estGyroscope']['gyro.y'])

    ax[2].plot(np.degrees(data_usd['fixedFrequency']['stateEstimateZ.rateYaw'] / 1000))

    plt.show()

