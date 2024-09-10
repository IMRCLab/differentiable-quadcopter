import torch
import numpy as np
import pandas as pd
from controller import ControllerLeeKhaled
from quadrotor_pytorch import QuadrotorAutograd
from plot_figure8 import f, fdot, fdotdot, fdotdotdot
from torch import optim
from torch.nn import L1Loss

import matplotlib.pyplot as plt

# Simulation/Training loop for the Lee Controller
# 1. Create Lee controller instance with random gains
# 2. Compute desired trajectory to track
# 3. Initialize UAV at start of the trajectory
# 4. Track the trajectory using simulation

if __name__=="__main__":
    figure8_data = pd.read_csv('figure8.csv')
    # compute the final arrival time
    t_final = np.sum(figure8_data['duration'])
    dt = 1/50 # 50hz control frequency
    ts = np.arange(0, t_final, dt)
    # 2. compute the desired trajectory
    f_t = np.concat([f(figure8_data, t) for t in ts], axis=1).T
    fdot_t = np.concat([fdot(figure8_data, t) for t in ts], axis=1).T
    fdotdot_t = np.concat([fdotdot(figure8_data, t) for t in ts], axis=1).T
    fdotdotdot_t = np.concat([fdotdotdot(figure8_data, t) for t in ts], axis=1).T
    xd = f_t[:,0]
    yd = f_t[:,1]
    # initialize UAV
    quadrotor = QuadrotorAutograd()
    quadrotor.m = quadrotor.mass
    quadrotor.I = torch.diag(quadrotor.J.clone().detach())
    quadrotor.B0 = quadrotor.B0
    quadrotor.dt = dt
    # initialize controller instance
    controller = ControllerLeeKhaled(uavModel=quadrotor, kp=9.0, kv=7.0, kr=0.0055, kw=0.0013)
    # initialize optimizer for controller gains
    optimizer = optim.Adam(controller.parameters(), lr=5e-5)
    loss_fn = L1Loss()
    # 4. track the trajectory using simulation
    num_epochs = 50
    visualize_trajectory = False

    # initialize state for t=0
    # structure for the state vector of the UAV is:
    # s = [x, y, z, v_x, v_y, v_z, qw, qx, qy, qz, omega_roll ,omega_yaw, omega_pitch]

    # structure for the setpoint vector of the trajectory is:
    # desired_setpoint = [p_x, p_y, p_z, v_x, v_y, v_z, a_x, a_y, a_z, j_x, j_y, j_z, yaw]
    # training loop
    for epoch in range(num_epochs):
        # set up initial state
        # s_0 = [s.x, s.y, s.z, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        s = torch.tensor([f_t[0,0], f_t[0,1], f_t[0,2], 0, 0, 0, 1., 0, 0, 0, 0, 0, 0], dtype=torch.double)

        # collect true x,y positions for plotting and evaluation
        x = [s[0]]
        y = [s[1]]

        # simulate trajectory over time
        losses = []
        for i,t in enumerate(ts):
            # if epoch < 10 and i > 5:
            #     break
            # elif epoch < 20 and i > 20:
            #     break
            # elif epoch < 30 and i > 30:
            #     break
            # elif epoch < 40 and i > 40:
            #     break
            # elif epoch < 50 and i > 200:
            #     break
             ## construct current setpoint
            s_d = torch.tensor([f_t[i,0], f_t[i,1], f_t[i,2], fdot_t[i,0], fdot_t[i,1], fdot_t[i,2], fdotdot_t[i,0], fdotdot_t[i,1], fdotdot_t[i,2], fdotdotdot_t[i,0], fdotdotdot_t[i,1], fdotdotdot_t[i,2],f_t[i,3]], dtype=torch.double)

            # compute controls
            thrustSI, torque, desW = controller(s.flatten(), s_d)

            # TODO: compute forces for rotors from thrust (1x1) and torque (3x1)
            force = quadrotor.B0.inverse() @ torch.concat([thrustSI, torque])

            # simulate UAV
            s = quadrotor.step(s.reshape((1,-1)), force.reshape(1,4))

            # store x,y position for plotting
            x.append(s[0,0].item())
            y.append(s[0,1].item())

            # compute loss
            # TODO: change this to also compute the error in attitude
            loss = loss_fn(s_d[:3], s[0,:3])
            losses.append(loss)

        if visualize_trajectory:
            plt.plot(x,y, label='simulated trajectory')
            plt.plot(xd, yd, label='desired trajectory')
            plt.legend()
            plt.show()
        loss = torch.mean(torch.stack(losses))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")
        print(f"New gains: kp={controller.kp}, kw={controller.kw}, kv={controller.kv}, kr={controller.kr}")
