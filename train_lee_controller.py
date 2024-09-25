import torch
from torch import nn
import roma
import numpy as np
import pandas as pd
from controller_pytorch import ControllerLee
from quadrotor_pytorch import QuadrotorAutograd
from plot_figure8 import f, fdot, fdotdot, fdotdotdot
from trajectories import spline_segment, f, fdot, fdotdot, fdotdotdot
from torch import optim
from train_system_id import qsym_distance
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


class QuadrotorControllerModule(nn.Module):
    def __init__(self, dt, kp=1., kv=1., kw=1., kr=1.):
        super(QuadrotorControllerModule, self).__init__()
        self.quadrotor = QuadrotorAutograd()
        self.quadrotor.dt = dt
        self.controller = ControllerLee(uavModel=self.quadrotor, kp=kp, kv=kv, kw=kw, kr=kr)

        self.kp = nn.Parameter(torch.tensor(self.controller.kp).clone().detach().requires_grad_(True))
        self.kv = nn.Parameter(torch.tensor(self.controller.kv).clone().detach().requires_grad_(True))
        self.kw = nn.Parameter(torch.tensor(self.controller.kw).clone().detach().requires_grad_(True))
        self.kr = nn.Parameter(torch.tensor(self.controller.kr).clone().detach().requires_grad_(True))

        self.controller.kp = self.kp
        self.controller.kv = self.kv
        self.controller.kw = self.kw
        self.controller.kr = self.kr

        self.double()
    
    def forward(self, setpoints, s_0=None):
        """
        Simulate a trajectory along a list of setpoints.

        Parameters:
        -----------
        setpoints: torch.Tensor
            batch of setpoints with shape (B,T,S) where B is the batch size, T is the trajectory length and S is the setpoint dimension.

        s_0: torch.Tensor (optional)
            tensor with initial states with shape (B,X) where B is the batch size and X is the state dimension.

        Returns:
        --------
        y: torch.Tensor
            Tensor with actual states encountered during simulation with shape (B,T,X)
        Rds: torch.Tensor
            Tensor with desired rotations in rotation matrix representation.
        desWs: torch.Tensor
            Tensor with desired attitude rates.
        """
        if s_0 is not None:
            current_state = s_0
        else:
            # construct the initial states from the first setpoints if no initial state is given
            batch_size = setpoints.shape[1] # time first, batch second
            current_state = torch.zeros((batch_size, 13), dtype=setpoints.dtype)
            current_state[:,0] = setpoints[0,:,0]
            current_state[:,1] = setpoints[0,:,1]
            current_state[:,2] = setpoints[0,:,2]
            current_state[:,3] = setpoints[0,:,3]
            current_state[:,4] = setpoints[0,:,4]
            current_state[:,5] = setpoints[0,:,5]
            current_state[:,6] = 1.  # unit quaternion with identity rotation
        y = []
        Rds = []
        desWs = []
        for setpoint in setpoints:
            y += [current_state]
            thrustSI, torque, Rd, desW = self.controller.compute_controls(current_state=current_state, setpoint=setpoint)
            Rds += [Rd]
            desWs += [desW]
            force = self.quadrotor.B0.inverse() @ torch.concat([thrustSI, torque], dim=1)
            current_state = self.quadrotor.step(state=current_state, force=force.squeeze(-1))
        return torch.stack(y, dim=0), torch.stack(Rds, dim=0), torch.stack(desWs, dim=0) # time first in -> time first out

class QuadrotorControllerLoss(nn.Module):
    def __init__(self, loss_fn='L1'):
        super().__init__()
        if loss_fn == 'L1':
            self.loss_fn = nn.L1Loss()
        elif loss_fn == 'MSE':
            self.loss_fn = nn.MSELoss()

    def forward(self, states, setpoints, desR, desW):
        position_loss = self.loss_fn(states[...,0:3], setpoints[...,0:3])
        velocity_loss = self.loss_fn(states[...,3:6], setpoints[...,3:6])
        rotational_loss = torch.mean(qsym_distance(roma.rotmat_to_unitquat(desR), states[...,6:10]))
        omega_loss = self.loss_fn(desW, states[...,10:13])
        return (position_loss, velocity_loss, rotational_loss, omega_loss)
        

class NthOrderTrajectoryDataset(Dataset):
    def __init__(self, parameter_file, funcs, dt, transform, t_max=None, t_0=0, window_size=None, as_setpoints=True):
        parameters = pd.read_csv(parameter_file)
        duration = np.sum(parameters['duration'])
        if t_max is None:
            t_max = duration
        assert t_max <= duration, "t_max cannot be larger than the duration of the trajectory"
        # create grid for time axis
        self.ts = np.arange(t_0, t_max, dt)
        self.as_setpoints = as_setpoints
        self.order = len(funcs)

        # generate the data
        self.trajectory_data = spline_segment(funcs=funcs, coeffs=parameters, ts=self.ts)
        self.transform = transform
        self.window_size = window_size
        if self.window_size is None:
            self.window_size = len(self.ts)
        self.slice_into_windows(window_size=self.window_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        window = self.data[idx]
        if self.as_setpoints:
            window = self.to_setpoint(window)
        if self.transform:
            window = self.transform(window)
        return window

    # x: input to the neural network
    # y: target output of the neural network
    # => x=initial position; y=desired trajectory
    def slice_into_windows(self, window_size):
        """
        Slices the dataset into windows of a given length.
        """
        # return Nx4xwindow_size
        self.window_size = window_size
        self.data = sliding_window_view(x=self.trajectory_data, window_shape=self.window_size, axis=0).transpose((0,3,1,2))
    
    def to_setpoint(self, window):
        return np.concat([window[...,0,0:3], window[...,1,0:3], window[...,2,0:3], window[...,3,0:3], window[...,0,3:4]], axis=-1)

    
def train_quadrotor_controller_module(model, criterion, optimizer, trainloader):
    running_loss = 0.0
    pbar = tqdm(total=len(trainloader))
    model.train()
    for i, setpoints in enumerate(trainloader):
        optimizer.zero_grad()
        states, Rds, desWs = model(setpoints.transpose(0,1))  # seq_len x batch_size x state_size
        position_loss, velocity_loss, rotational_loss, omega_loss = criterion(states.transpose(0,1), setpoints, Rds.transpose(0,1), desWs.squeeze(-1).transpose(0,1))
        loss = position_loss + velocity_loss + rotational_loss + omega_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_description(f"loss={running_loss / (i+1):0.4g}")
        pbar.update(1)
    pbar.close()

def run_trajectory(model, trajectory, s_0=None):
    model.eval()
    with torch.no_grad():
        states, Rds, desWs = model(trajectory.unsqueeze(1), s_0=s_0)
    return states.squeeze(1)



if __name__=="__main__":
    # figure8_data = pd.read_csv('figure8.csv')
    dt = 1/50 # 50hz control frequency
    lr = 1e-3
    epochs = 10
    visualize_trajectory = True
    report_gains = True

    # write custom transform to create setpoint
    figure8_dataset = NthOrderTrajectoryDataset('figure8.csv', [f, fdot, fdotdot, fdotdotdot], dt=dt, transform=torch.tensor)
    figure8_dataset.slice_into_windows(5)

    train_dataloader = DataLoader(figure8_dataset, batch_size=8, shuffle=True)
    quadrotor_controller_module = QuadrotorControllerModule(dt=dt, kp=9., kv=7., kw=0.0013, kr=0.0055)
    criterion = QuadrotorControllerLoss(loss_fn='L1')
    optimizer = optim.Adam(quadrotor_controller_module.parameters(), lr=lr)

    for epoch in range(epochs):
        train_quadrotor_controller_module(model=quadrotor_controller_module, criterion=criterion, optimizer=optimizer, trainloader=train_dataloader)
        if report_gains:
            print(f"Gains after {epoch+1} epochs:\nkp={quadrotor_controller_module.kv.item()}\tkv={quadrotor_controller_module.kv.item()}\tkw={quadrotor_controller_module.kw.item()}\tkr={quadrotor_controller_module.kr.item()}")

    if visualize_trajectory:
        ax = plt.figure().add_subplot(projection='3d')
        setpoint_trajectory = torch.tensor(figure8_dataset.to_setpoint(figure8_dataset.trajectory_data), dtype=torch.double)
        states = run_trajectory(quadrotor_controller_module, setpoint_trajectory)
        ax.plot(states[:,0], states[:,1], states[:,2], label='simulated trajectory')
        ax.plot(setpoint_trajectory[:,0],setpoint_trajectory[:,1], setpoint_trajectory[:,2], label='desired trajectory')
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=-35, roll=0)
        plt.show()
