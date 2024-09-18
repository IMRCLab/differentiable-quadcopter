import torch
from torch import nn
import roma
import numpy as np
import pandas as pd
from controller import ControllerLeeKhaled
from quadrotor_pytorch import QuadrotorAutograd
from plot_figure8 import f, fdot, fdotdot, fdotdotdot
from torch import optim
from torch.nn import L1Loss
from train_system_id import qsym_distance

from torch.utils.data import Dataset

import matplotlib.pyplot as plt


class QuadrotorControllerModule(nn.Module):
    def __init__(self, dt, planning_horizon):
        super(QuadrotorControllerModule, self).__init__()
        self.quadrotor = QuadrotorAutograd()
        self.quadrotor.dt = dt
        self.controller = ControllerLeeKhaled(uavModel=self.quadrotor)

        self.kp = nn.Parameter(torch.tensor(self.controller.kp))
        self.kv = nn.Parameter(torch.tensor(self.controller.kv))
        self.kw = nn.Parameter(torch.tensor(self.controller.kw))
        self.kr = nn.Parameter(torch.tensor(self.controller.kr))

        self.controller.kp = self.kp
        self.controller.kv = self.kv
        self.controller.kw = self.kw
        self.controller.kr = self.kr

        self.planning_horizon = planning_horizon

        self.double()
    
    def forward(self, s_d, s_0=None):
        if s_0 is not None:
            s = s_0
        else:
            s = s_d[0]
        y = [s]
        for i in self.planning_horizon:
            control = self.controller.computeControl()
            s = self.quadrotor.step()
            y += [s]
        return torch.stack(y)

        

# Simulation/Training loop for the Lee Controller
# 1. Create Lee controller instance with random gains
# 2. Compute desired trajectory to track
# 3. Initialize UAV at start of the trajectory
# 4. Track the trajectory using simulation
class FourthOrderTrajectoryDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __get_item__(self):
        pass
    # x: input to the neural network
    # y: target output of the neural network
    # => x=initial position; y=desired trajectory
    def slice_into_windows(self, window_size):
        """
        Slices the dataset into windows of a given length.
        """
        # return Nx4xwindow_size
    pass

if __name__=="__main__":
    figure8_data = pd.read_csv('figure8.csv')
    # compute the final arrival time
    t_final = np.sum(figure8_data['duration'])
    dt = 1/100 # 50hz control frequency
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
    quadrotor.dt = dt
    # initialize controller instance
    controller = ControllerLeeKhaled(uavModel=quadrotor, kp=9.0, kv=7.0, kr=0.0055, kw=0.0013)
    # controller = ControllerLeeKhaled(uavModel=quadrotor, kp=8.9986, kv=7.0014, kr=0.0069, kw=0.0004)
    # controller = ControllerLeeKhaled(uavModel=quadrotor, kp=1.0, kv=1.0, kr=.01, kw=.01)
    # initialize optimizer for controller gains
    optimizer = optim.Adam(controller.parameters(), lr=1e-5)
    loss_fn = L1Loss()
    # 4. track the trajectory using simulation
    num_epochs = 50
    visualize_trajectory = True

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
            ## construct current setpoint
            s_d = torch.tensor([f_t[i,0], f_t[i,1], f_t[i,2], fdot_t[i,0], fdot_t[i,1], fdot_t[i,2], fdotdot_t[i,0], fdotdot_t[i,1], fdotdot_t[i,2], fdotdotdot_t[i,0], fdotdotdot_t[i,1], fdotdotdot_t[i,2],f_t[i,3]], dtype=torch.double)

            # compute controls
            thrustSI, torque, desR, desW = controller(s.flatten(), s_d)

            # compute forces for rotors from thrust (1x1) and torque (3x1)
            force = quadrotor.B0.inverse() @ torch.concat([thrustSI, torque])

            # simulate UAV
            s = quadrotor.step(s.reshape((1,-1)), force.reshape(1,4))

            # store x,y position for plotting
            x.append(s[0,0].item())
            y.append(s[0,1].item())

            # compute loss
            # TODO: change this to also compute the error in attitude
            position_loss = loss_fn(s_d[:3], s[0,:3])
            velocity_loss = loss_fn(s_d[3:6], s[0,3:6])
            # rotational_loss = qsym_distance(roma.rotmat_to_unitquat(desR), s[0,6:10])
            omega_loss = loss_fn(desW.flatten(), s[0,10:13])
            losses.append(position_loss+velocity_loss+omega_loss)

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

