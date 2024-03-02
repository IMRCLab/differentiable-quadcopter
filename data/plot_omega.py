import argparse

import matplotlib.pyplot as plt
import numpy as np
import rowan

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

    # T = len(data_usd['fixedFrequency']['timestamp'])
    # k2 = 0
    # gyro = []
    # for k1 in range(T-1):
    #     t_tF = data_usd['fixedFrequency']['timestamp'][k1+1]
    #     gyro_data = []
    #     while k2 < len(data_usd['estGyroscope']['timestamp']):
    #         t_eG = data_usd['estGyroscope']['timestamp'][k2]
    #         gyro_data.append(data_usd['estGyroscope']['gyro.x'][k2])
    #         if t_eG > t_tF:
    #             break
    #         k2 = k2 + 1
    #     print(gyro_data)
    #     gyro.append(np.mean(gyro_data))

    quats = []
    for q_compressed in data_usd['fixedFrequency']['stateEstimateZ.quat']:
        # q is in [x,y,z,w] format
        q = cfusdlog.quatdecompress(q_compressed)
        q_rowan = np.array([q[3], q[0], q[1], q[2]])
        quats.append(q_rowan)
    quats = np.array(quats)

    # try to numerically estimate omega
    est_omega = []
    est_omega_time = []
    t1 = 0
    while t1 < quats.shape[0]-1:
        q1 = quats[t1]
        t2 = t1 + 1
        while t2 < quats.shape[0]-1:
            if not np.allclose(quats[t1], quats[t2], rtol=0.2):
                dt = time_fF[t2] - time_fF[t1]
                # see https://math.stackexchange.com/questions/2282938/converting-from-quaternion-to-angular-velocity-then-back-to-quaternion
                omega = 2 * rowan.multiply(rowan.conjugate(quats[t1]), quats[t2])[1:4] / dt
                est_omega.append(omega)
                est_omega_time.append(time_fF[t1])
                break
            t2 += 1
        t1 = t2

    est_omega = np.array(est_omega)
    est_omega_time = np.array(est_omega_time)

    rpy = rowan.to_euler(quats, 'xyz')

    fig, ax = plt.subplots(2, 3, sharex='all')

    ax[0,0].plot(time_fF, np.degrees(rpy[:,0]))
    ax[0,1].plot(time_fF, np.degrees(rpy[:,1]))
    ax[0,2].plot(time_fF, np.degrees(rpy[:,2]))
    ax[1,0].set_ylabel("rotation [deg]")


    ax[1,0].plot(time_fF, np.degrees(data_usd['fixedFrequency']['stateEstimateZ.rateRoll'] / 1000))
    ax[1,0].plot(time_eG, data_usd['estGyroscope']['gyro.x'])
    ax[1,0].plot(est_omega_time, np.degrees(est_omega[:,0]))

    # ax[0].plot(time_fF[1:], gyro)

    ax[1,1].plot(time_fF, np.degrees(data_usd['fixedFrequency']['stateEstimateZ.ratePitch'] / -1000))
    ax[1,1].plot(time_eG, data_usd['estGyroscope']['gyro.y'])
    ax[1,1].plot(est_omega_time, np.degrees(est_omega[:,1]))

    ax[1,2].plot(time_fF, np.degrees(data_usd['fixedFrequency']['stateEstimateZ.rateYaw'] / 1000))
    ax[1,2].plot(time_eG, data_usd['estGyroscope']['gyro.z'])
    ax[1,2].plot(est_omega_time, np.degrees(est_omega[:,2]))


    ax[1,0].set_ylabel("angular velocity [deg/s]")
    ax[1,0].set_xlabel("time [s]")

    plt.show()

