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

    T = len(data_usd['fixedFrequency']['timestamp'])

    fig, ax = plt.subplots(5, 1)
    ax[0].plot(data_usd['fixedFrequency']['rpm.m1'])
    ax[1].plot(data_usd['fixedFrequency']['rpm.m2'])
    ax[2].plot(data_usd['fixedFrequency']['rpm.m3'])
    ax[3].plot(data_usd['fixedFrequency']['rpm.m4'])

    # estimate thrust from rpm data
    kf = 2.1114e-10

    thrust = kf * (np.power(data_usd['fixedFrequency']['rpm.m1'], 2) + \
        np.power(data_usd['fixedFrequency']['rpm.m2'], 2) + \
        np.power(data_usd['fixedFrequency']['rpm.m3'], 2) + \
        np.power(data_usd['fixedFrequency']['rpm.m4'], 2))

    # estimate thrust from pwm data
    vbat_norm = data_usd['fixedFrequency']['pm.vbatMV'] / 1000 / 4.2

    thrust2 = pwm2force(data_usd['fixedFrequency']['pwm.m1_pwm'] / 65536, vbat_norm) + \
        pwm2force(data_usd['fixedFrequency']['pwm.m2_pwm'] / 65536, vbat_norm) + \
        pwm2force(data_usd['fixedFrequency']['pwm.m3_pwm'] / 65536, vbat_norm) + \
        pwm2force(data_usd['fixedFrequency']['pwm.m4_pwm'] / 65536, vbat_norm)

    ax[4].plot(thrust / 9.81 * 1000) # grams
    ax[4].plot(thrust2 / 9.81 * 1000)  # grams
    # ax[4].plot(data_usd['fixedFrequency']['asc37800.v_mV'])  # grams
    # ax[4].plot(data_usd['fixedFrequency']['pm.vbatMV'])  # grams

    # ax[4].plot(data_usd['fixedFrequency']['stateEstimateZ.az'] / 1000 * 0.0388 / 9.81 * 1000) # m/s^2
    # print(np.mean(thrust / 9.81 * 1000))

    plt.show()

