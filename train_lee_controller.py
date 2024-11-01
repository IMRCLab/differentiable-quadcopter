import argparse
import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from distutils.util import strtobool
from torch import nn
import roma
import numpy as np
import pandas as pd
from controller_pytorch import ControllerLee
from quadrotor_pytorch import QuadrotorAutograd
from trajectories import spline_segment, f, fdot, fdotdot, fdotdotdot
from torch import optim
from train_system_id import qsym_distance
from tqdm import tqdm
from controller_pytorch import vee_so3

from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


class QuadrotorControllerModule(nn.Module):
    def __init__(self, dt, kp=[[1.],[1.],[1.]], kv=[[1.],[1.],[1.]], kw=[[1.],[1.],[1.]], kr=[[1.],[1.],[1.]], mass=None, inertia=None, noise_on=False):
        super().__init__()
        self.quadrotor = QuadrotorAutograd(noise_on=noise_on)
        self.quadrotor.dt = dt

        # If mass and inertia are given initialize controller accordingly
        if mass is not None and inertia is not None:
            self.controller = ControllerLee(kp=kp, kv=kv, kw=kw, kr=kr, mass=mass, inertia=inertia)
            self.mass = nn.Parameter(self.controller.m.clone().detach().requires_grad_(True))
            self.inertia = nn.Parameter(self.controller.I.clone().detach().requires_grad_(True))
        else:
            self.controller = ControllerLee(kp=kp, kv=kv, kw=kw, kr=kr, mass=self.quadrotor.m, inertia=self.quadrotor.I)
            self.mass = nn.Parameter(self.controller.m.clone().detach(), requires_grad=False)
            self.inertia = nn.Parameter(self.controller.I.clone().detach(), requires_grad=False)

        self.controller.m = self.mass
        self.controller.I = self.inertia

        self.kp = nn.Parameter(self.controller.kp.clone().detach().requires_grad_(True))
        self.kv = nn.Parameter(self.controller.kv.clone().detach().requires_grad_(True))
        self.kw = nn.Parameter(self.controller.kw.clone().detach().requires_grad_(True))
        self.kr = nn.Parameter(self.controller.kr.clone().detach().requires_grad_(True))

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
            batch of setpoints with shape (T,B,S) where B is the batch size, T is the trajectory length and S is the setpoint dimension.

        s_0: torch.Tensor (optional)
            tensor with initial states with shape (B,X) where B is the batch size and X is the state dimension. State format is [x,y,z, v_x, v_y, v_z, q_w, q_x, q_y, q_z, omega_x, omega_y, omega_z]

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
        # rotational_loss = torch.mean(qsym_distance(roma.rotmat_to_unitquat(desR), states[...,6:10]))
        R = roma.unitquat_to_rotmat(states[..., 6:10])
        er = 0.5 * vee_so3(desR.transpose(-2,-1) @ R - R.transpose(-2,-1) @ desR) # tracking error in rotation
        rotational_loss = torch.mean(er**2)
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
        dataset_size = len(self.trajectory_data)
        if window_size > dataset_size:
            window_size = dataset_size
        self.window_size = window_size
        N = dataset_size // window_size
        splits = np.split(self.trajectory_data[:N*window_size], N, axis=0)
        self.data = np.stack(splits, axis=0) # num_windows x window_size x num_spline_order x pose_size
    
    def to_setpoint(self, window):
        # setpoint format: [x,y,z, v_x, v_y, v_z, a_x, a_y, a_z, jerk_x, jerk_y, jerk_z, yaw]
        return np.concat([window[...,0,0:3], window[...,1,0:3], window[...,2,0:3], window[...,3,0:3], window[...,0,3:4]], axis=-1)

    
def train_quadrotor_controller_module(model, criterion, optimizer, trainloader, writer=None, clip_gradient_norm=1.0):
    running_loss, running_position_loss, running_velocity_loss, running_rotational_loss, running_omega_loss = 0.0, 0.0, 0.0, 0.0, 0.0

    pbar = tqdm(total=len(trainloader))
    model.train()
    for i, setpoints in enumerate(trainloader):
        optimizer.zero_grad()
        states, Rds, desWs = model(setpoints.transpose(0,1))  # seq_len x batch_size x state_size
        position_loss, velocity_loss, rotational_loss, omega_loss = criterion(states.transpose(0,1), setpoints, Rds.transpose(0,1), desWs.squeeze(-1).transpose(0,1))
        loss = position_loss + velocity_loss + rotational_loss + omega_loss
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                with torch.no_grad():
                    parameter.grad = torch.clamp(parameter.grad, -clip_gradient_norm*torch.abs(parameter), clip_gradient_norm*torch.abs(parameter))
        optimizer.step()

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.clamp_(min=1e-8)

        running_loss += loss.item()
        running_position_loss += position_loss.item()
        running_velocity_loss += velocity_loss.item()
        running_rotational_loss += rotational_loss.item()
        running_omega_loss += omega_loss.item()
        pbar.set_description(f"loss={running_loss / (i+1):0.4g}")
        pbar.update(1)
    pbar.close()
    return running_position_loss / (i+1), running_velocity_loss / (i+1), running_rotational_loss / (i+1), running_omega_loss / (i+1)

def run_trajectory(model, trajectory, s_0=None):
    model.eval()
    with torch.no_grad():
        states, Rds, desWs = model(trajectory.unsqueeze(1), s_0=s_0)
    return states.squeeze(1), Rds.squeeze(1), desWs.squeeze()



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameter-file', type=str, default='figure8.csv',
                        help='name of the file which contains the parameters for the trajectory splines')
    parser.add_argument('--dt', type=float, default=1/100,
                        help='duration of a simulation step in seconds')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='the learning rate of the optimizer')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='the size of the batches for training')
    parser.add_argument('--epochs', type=int, default=500,
                        help='the number of epochs to run the optimization')
    parser.add_argument('--window-size', type=int, default=3,
                        help='the length of the time windows the trajectory is cut into for training')
    parser.add_argument('--visualize-trajectory', type=lambda x: bool(strtobool(x)) ,default=True,
                        help='Toggles whether or not the trajectory should be visualized after training')
    parser.add_argument('--report-gains', type=lambda x: bool(strtobool(x)) ,default=False,
                        help='Toggles whether or not the current gains are reported during training')
    parser.add_argument('--track', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='the optimizer used tune the gains')
    parser.add_argument('--gain-init', type=str, default='ones',
                        help='the initial gains used for tuning')
    parser.add_argument('--loss-fn', type=str, default='MSE',
                        help='the loss function used for the optimization')
    parser.add_argument('--sim-noise', type=lambda x: bool(strtobool(x)), default=False,
                        help='if toggled, the simulation will be run with noise')
    parser.add_argument('--random-init', type=lambda x: bool(strtobool(x)), default=False,
                        help='if toggled, the simulation will sample random gains in the beginning')
    parser.add_argument('--double-window-size-on-plateau', type=lambda x: bool(strtobool(x)), default=True,
                        help='if toggled, the window size will double if the training loss does not decrease any more')
    parser.add_argument('--num-epochs-plateau', type=int, default=50,
                        help='number of epochs to wait on plateau before window size is doubled')
    parser.add_argument('--model-checkpoint-file', type=str, default=None,
                        help='if provided the model parameters are loaded from this file')
    parser.add_argument('--save-checkpoint-file', type=str, default=None,
                        help='if provided the modle parameters are saved to this file')
    parser.add_argument('--position-weight', type=float, default=1.,
                        help='weight for the position loss')
    parser.add_argument('--velocity-weight', type=float, default=1.,
                        help='weight for the velocity loss')
    parser.add_argument('--rotational-weight', type=float, default=1.,
                        help='weight for the rotational loss')
    parser.add_argument('--omega-weight', type=float, default=1.,
                        help='weight for the omega loss')
    parser.add_argument('--clip-gradient-norm', type=float, default=1.0,
                        help='gradient norm to clip the gradient at')
    args = parser.parse_args()
    
    # write custom transform to create setpoint
    figure8_dataset = NthOrderTrajectoryDataset('figure8.csv', [f, fdot, fdotdot, fdotdotdot], dt=args.dt, transform=torch.tensor)
    figure8_dataset.slice_into_windows(window_size=args.window_size)

    train_dataloader = DataLoader(figure8_dataset, batch_size=args.batch_size, shuffle=True)
    if args.gain_init == 'random':
        kp = np.random.uniform(1., 10.)
        kv = np.random.uniform(1., 10.)
        kr = np.random.uniform(1e-8, 1.)
        kw = np.random.uniform(1e-8, 1.)
        quadrotor_controller_module = QuadrotorControllerModule(dt=args.dt, kp=[[kp],[kp],[kp]], kv=[[kv],[kv],[kv]], kw=[[kw],[kw],[kw]], kr=[[kr],[kr],[kr]], noise_on=args.sim_noise)
    elif args.gain_init == 'pretuned':
        quadrotor_controller_module = QuadrotorControllerModule(dt=args.dt, kp=[[9.],[9.],[9.]], kv=[[7.],[7.],[7.]], kw=[[0.0013],[0.0013],[0.0013]], kr=[[0.0055],[0.0055],[0.0055]], noise_on=args.sim_noise)
    elif args.gain_init == 'ones':
        quadrotor_controller_module = QuadrotorControllerModule(dt=args.dt, kp=[[1.],[1.],[1.]], kv=[[1.],[1.],[1.]], kw=[[1.],[1.],[1.]], kr=[[1.],[1.],[1.]], noise_on=args.sim_noise)
    else:
        ValueError(f'Gain initialization {args.gain_init} is not supported')

    criterion = QuadrotorControllerLoss(loss_fn=args.loss_fn)
    if args.optimizer == 'SGD':
        optimizer = optim.SGD
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam
    elif args.optimizer == 'LBFGS':
        optimizer = optim.LBFGS
    else:
        raise ValueError(f'Optimizer {args.optimizer} is not a valid option')

    optimizer = optimizer(quadrotor_controller_module.parameters(), lr=args.lr)

    if args.save_checkpoint_file:
        run_name = args.save_checkpoint_file
        os.makedirs("checkpoints", exist_ok=True)
    else:
        run_name = f'lee_controller__{int(time.time())}'

    if args.track:
        writer = SummaryWriter(f"runs/parameters_{run_name}")
    else:
        writer = None

    if args.double_window_size_on_plateau:
        best_loss = torch.inf
        iterations_since_decrease = 0
    
    if args.model_checkpoint_file:
        checkpoint = torch.load(f'checkpoints/{args.model_checkpoint_file}', weights_only=True)
        quadrotor_controller_module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        args.window_size = checkpoint['window_size']
        figure8_dataset.slice_into_windows(args.window_size)
        print(f'Resuming training of {args.model_checkpoint_file} at epoch {start_epoch}')
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        position_loss, velocity_loss, rotational_loss, omega_loss = train_quadrotor_controller_module(model=quadrotor_controller_module, criterion=criterion, optimizer=optimizer, trainloader=train_dataloader, writer=writer, clip_gradient_norm=args.clip_gradient_norm)
        loss = args.position_weight * position_loss + args.velocity_weight * velocity_loss + args.rotational_weight * rotational_loss + args.omega_weight * omega_loss

        if writer is not None:
            writer.add_scalar("loss/position_loss", position_loss, global_step=epoch)
            writer.add_scalar("loss/velocity_loss", velocity_loss, global_step=epoch)
            writer.add_scalar("loss/rotational_loss", rotational_loss, global_step=epoch)
            writer.add_scalar("loss/omega_loss", omega_loss, global_step=epoch)
            writer.add_scalar("parameters/window_size", args.window_size, global_step=epoch)
            writer.add_scalars("gains/kp", {'x': quadrotor_controller_module.kp[0], 'y': quadrotor_controller_module.kp[1], 'z': quadrotor_controller_module.kp[2]}, global_step=epoch)
            writer.add_scalars("gains/kv", {'x': quadrotor_controller_module.kv[0], 'y': quadrotor_controller_module.kv[1], 'z': quadrotor_controller_module.kv[2]}, global_step=epoch)
            writer.add_scalars("gains/kr", {'x': quadrotor_controller_module.kr[0], 'y': quadrotor_controller_module.kr[1], 'z': quadrotor_controller_module.kr[2]}, global_step=epoch)
            writer.add_scalars("gains/kw", {'x': quadrotor_controller_module.kw[0], 'y': quadrotor_controller_module.kw[1], 'z': quadrotor_controller_module.kw[2]}, global_step=epoch)
            # writer.add_scalar("physical_parameters/mass", quadrotor_controller_module.mass, global_step=epoch)
            # writer.add_scalar("physical_parameters/inertia", quadrotor_controller_module.inertia, global_step=epoch)

        if args.double_window_size_on_plateau:
            if loss < best_loss - 1e-4:
                best_loss = loss
                iterations_since_decrease = 0
            else:
                iterations_since_decrease += 1
            
            if iterations_since_decrease > args.num_epochs_plateau:
                args.window_size *= 2
                if args.window_size > 350:
                    args.window_size = 350
                figure8_dataset.slice_into_windows(window_size=args.window_size)
                iterations_since_decrease = 0
                best_loss = torch.inf

        if args.save_checkpoint_file is not None: 
            # save gains/model
            torch.save({'epoch': epoch,
                        'model_state_dict': quadrotor_controller_module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'window_size': args.window_size,
                        }, f'checkpoints/{run_name}.pt')

        if args.report_gains:
            with torch.no_grad():
                print(f"Gains after {epoch+1} epochs:\nkp={quadrotor_controller_module.kp.tolist()}\tkv={quadrotor_controller_module.kv.tolist()}\tkw={quadrotor_controller_module.kw.tolist()}\tkr={quadrotor_controller_module.kr.tolist()}")

    if writer is not None: 
        writer.close()

    if args.visualize_trajectory or args.plot_errors:
        setpoint_trajectory = torch.tensor(figure8_dataset.to_setpoint(figure8_dataset.trajectory_data), dtype=torch.double)
        states, Rds, desWs = run_trajectory(quadrotor_controller_module, setpoint_trajectory)

    if args.visualize_trajectory:
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(states[:,0], states[:,1], states[:,2], label='simulated trajectory')
        ax.plot(setpoint_trajectory[:,0],setpoint_trajectory[:,1], setpoint_trajectory[:,2], label='desired trajectory')
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=-35, roll=0)
        plt.show()
    