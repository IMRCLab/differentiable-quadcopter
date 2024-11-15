import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import roma
import time
import torch
from tqdm import tqdm

from data import cfusdlog
from distutils.util import strtobool
from torch.utils.data.dataset import TensorDataset, Dataset
from torch.utils.tensorboard import SummaryWriter
from quadrotor_pytorch import QuadrotorAutograd 
from utils import vee_so3, slice_dataset

from torch import nn
from torch import optim
from torch.utils.data import DataLoader


class QuadrotorSimulationModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.quadrotor = QuadrotorAutograd(**kwargs)

        # optimize mass
        self.mass = nn.Parameter(torch.tensor([self.quadrotor.m]))
        self.quadrotor.m = self.mass

        self.inertia = nn.Parameter(self.quadrotor.I)
        self.quadrotor.I = self.inertia

        # self.B0 = nn.Parameter(self.quad.B0)
        # self.quad.B0 = self.B0

        self.kf = 2.1
        self.double()
        # self.kf = nn.Parameter(torch.tensor([self.kf]))

    def forward(self, controls, s_0, dts=None):
        """
        Simulate a quadrotor for some given controls.

        Parameters:
        -----------
        motor_controls: torch.Tensor
            batch of motor controls with shape (T, B, S) where B is the batch size, T is the trajectory length and S is control dimension.
        s_0: torch.Tensor
            batch of initial states with shape (B, X) where B is the batch size and X is the state dimension
        """
        current_state = s_0
        states = [current_state]
        for i, control in enumerate(controls):
            force = self.kf * 1e-10 * torch.pow(control,2)  # motor controls to force
            if dts is not None:
                dt = dts[i]
            else:
                dt = self.quadrotor.dt
            current_state = self.quadrotor.step(state=current_state, force=force, dt=dt)
            states += [current_state]
        # state = x[:,0:13]
        # kf = 1e-10
        # print(self.kf, x)
        # force = self.kf * 1e-10 * torch.pow(x[:,13:], 2)  # motor controls to force
        # force = self.kf * x[0, 13:]
        # print(force)
        # exit()
        # print(x[0,13:], force)
        # exit()
        # print(torch.sum(force))
        # next_state = self.quad.step(state, force)

        # return next_state
        return torch.stack(states, dim=0)


# quaternion norm (adopted from rowan)
def qnorm(q):
    return torch.linalg.norm(q, dim=-1, keepdim=True)

# quaternion sym distance (adopted from rowan)
def qsym_distance(p, q):
    return torch.minimum(qnorm(p - q), qnorm(p + q))

class QuadrotorLoss(nn.Module):
    def __init__(self):
        super(QuadrotorLoss, self).__init__()

    def forward(self, input, target):
        # print(input, target)
        position_loss = torch.nn.functional.mse_loss(input[...,0:3], target[...,0:3])
        velocity_loss = torch.nn.functional.mse_loss(input[...,3:6], target[...,3:6])
        # transform quaternions to rotation matrices and compute error in SO(3)
        R_input = roma.unitquat_to_rotmat(input[...,6:10])
        R_target = roma.unitquat_to_rotmat(target[...,6:10])
        angle_errors = 0.5 * vee_so3(R_target.transpose(-2,-1) @ R_input - R_input.transpose(-2,-1) @ R_target)
        angle_loss = torch.mean(angle_errors)
        omega_loss = torch.nn.functional.mse_loss(input[...,10:13], target[...,10:13])
        return position_loss, velocity_loss, angle_loss, omega_loss

class FlightDataset(Dataset):
    def __init__(self, file_name, transform, window_size=None):
        super().__init__()
        self.raw_dts, self.raw_states, self.raw_controls = load_cfusd(file_name)
        self.window_size = window_size
        self.transform = transform
        if self.window_size is None:
            self.window_size = len(self.raw_states)
        self.slice_into_windows(window_size=self.window_size)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        states_window = self.states[idx]
        controls_window = self.controls[idx]
        dts_window = self.dts[idx]
        if self.transform:
            states_window = self.transform(states_window)
            controls_window = self.transform(controls_window)
        return dts_window, states_window, controls_window
    
    def slice_into_windows(self, window_size):
        """
        Slices the dataset into windows of a given length.
        """
        dataset_size = len(self.raw_states)
        if window_size > dataset_size:
            window_size = dataset_size
        self.window_size = window_size
        self.states = slice_dataset(self.raw_states, window_size=window_size)
        
        dataset_size = len(self.raw_dts)
        N = dataset_size // (window_size-1)  # number of segments
        dts_splits = np.split(self.raw_dts[:N*(window_size-1)], N, axis=0)
        self.dts = np.stack(dts_splits, axis=0)
        control_splits = np.split(self.raw_controls[:N*(window_size-1)], N, axis=0)
        self.controls = np.stack(control_splits, axis=0)

# def train_loop(dataloader, model: nn.Module, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     training_loss = 0

#     model.train()

#     for batch, (X, y) in enumerate(dataloader):
#         # Compute prediction and loss
#         pred = model(X)
#         loss = loss_fn(pred, y)

#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         clip_gradient_norm = 0.1
#         for parameter in model.parameters():
#             if parameter.grad is not None:
#                 with torch.no_grad():
#                     parameter.grad = torch.clamp(parameter.grad, -clip_gradient_norm*torch.abs(parameter), clip_gradient_norm*torch.abs(parameter))
#             print(parameter.grad)
#         optimizer.step()

#         training_loss += loss.item()

#         # if batch % 100 == 0:
#         #     loss, current = loss.item(), batch * len(X)
#         #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

#     training_loss /= size
#     print(f"Training Error: \n Avg loss: {training_loss:>8f} \n")
#     return training_loss


def train_quadrotor_system_identification(model, criterion, optimizer, trainloader, clip_gradient_norm=0.1):
    running_loss, running_position_loss, running_velocity_loss, running_angle_loss, running_omega_loss = 0.0, 0.0, 0.0, 0.0, 0.0

    pbar = tqdm(total=len(trainloader))
    model.train()
    for batch_idx, (recorded_dts, recorded_states, recorded_controls) in enumerate(trainloader):  # recorded states
        optimizer.zero_grad()
        simulated_states = model(s_0=recorded_states[:,0,:], controls=recorded_controls.transpose(0,1), dts=recorded_dts.transpose(0,1))
        position_loss, velocity_loss, angle_loss, omega_loss = criterion(simulated_states.transpose(0,1), recorded_states)
        loss = position_loss + velocity_loss + angle_loss + omega_loss
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                with torch.no_grad():
                    # new fancy gradient projection method
                    relative_grad = parameter.grad / parameter
                    r_max = 0.0
                    for r in relative_grad:
                        if torch.abs(r) > clip_gradient_norm:
                            r_max = torch.abs(r)
                        if r_max > 0.0:
                            relative_grad = relative_grad / r_max * clip_gradient_norm
                            parameter.grad = relative_grad * parameter
                    # parameter.grad = torch.clamp(parameter.grad, -clip_gradient_norm*torch.abs(parameter), clip_gradient_norm*torch.abs(parameter))
                    # print(parameter.grad)
        optimizer.step()

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.clamp_(min=1e-8)
        
        running_loss += loss.item()
        running_position_loss += position_loss.item()
        running_velocity_loss += velocity_loss.item()
        running_angle_loss += angle_loss.item()
        running_omega_loss += omega_loss.item()
        pbar.set_description(f"loss={running_loss / (batch_idx+1):0.4g}")
        pbar.update(1)
    pbar.close()
    return running_loss / (batch_idx+1), running_position_loss / (batch_idx+1), running_velocity_loss / (batch_idx+1), running_angle_loss / (batch_idx+1), running_omega_loss / (batch_idx+1)


def test_loop(dataloader, model: nn.Module, loss_fn):
    size = len(dataloader.dataset)
    test_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= size
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    return test_loss


# pwm normalized [0-1]; vbat normalized [0-1]
# output in N
def pwm2force(pwm, vbat):
    C_00 = 11.093358483549203
    C_10 = -39.08104165843915
    C_01 = -9.525647087583181
    C_20 = 20.573302305476638
    C_11 = 38.42885066644033
    return (C_00 + C_10*pwm + C_01*vbat + C_20*pwm**2 + C_11*vbat*pwm) / 1000 * 9.81


def load_cfusd(filename):
    """
    Load binary logged sensor data from the Crazyflie2. Returns a tensor with the state (position and quaternion rotation)
    and its derivative (translational and rotational velocity) and the quadrotor control command

    Parameters:
    -----------
    filename: str
        name of the file the sensor data should be loaded from
    
    Returns:
    --------
    dt: float
        Average timestep between states/controls in ms
    states: torch.Tensor
        Tensor containing the states and derivatives with shape (T, S) where T is the trajectory length, S is the state dimension
    controls: torch.Tensor
        Tensor containing the controls with shape (T, C) where T is the trajectory length and C is the control dimension
    """
    # decode binary log data
    data_usd = cfusdlog.decode(filename)

    T = len(data_usd['fixedFrequency']['timestamp'])

    dts = np.diff(data_usd['fixedFrequency']['timestamp']) / 1000.0
    dt = np.mean(dts)


    states = np.empty((T, 13), dtype=np.float64)
    controls = np.empty((T,4), dtype=np.float64)

    states[:, 0] = data_usd['fixedFrequency']['stateEstimateZ.x'] / 1000.0
    states[:, 1] = data_usd['fixedFrequency']['stateEstimateZ.y'] / 1000.0
    states[:, 2] = data_usd['fixedFrequency']['stateEstimateZ.z'] / 1000.0
    states[:, 3] = data_usd['fixedFrequency']['stateEstimateZ.vx'] / 1000.0
    states[:, 4] = data_usd['fixedFrequency']['stateEstimateZ.vy'] / 1000.0
    states[:, 5] = data_usd['fixedFrequency']['stateEstimateZ.vz'] / 1000.0
    for t in range(T):
        # q is in [x,y,z,w] format
        q = cfusdlog.quatdecompress(
            data_usd['fixedFrequency']['stateEstimateZ.quat'][t])
        # [w,x,y,z] format
        states[t, 6:7] = q[3:4]
        states[t, 7:10] = q[0:3]
    states[:, 10] = data_usd['fixedFrequency']['stateEstimateZ.rateRoll'] / 1000.0
    states[:, 11] = data_usd['fixedFrequency']['stateEstimateZ.ratePitch'] / 1000.0
    states[:, 12] = data_usd['fixedFrequency']['stateEstimateZ.rateYaw'] / 1000.0

    controls[:, 0] = data_usd['fixedFrequency']['rpm.m1']
    controls[:, 1] = data_usd['fixedFrequency']['rpm.m2']
    controls[:, 2] = data_usd['fixedFrequency']['rpm.m3']
    controls[:, 3] = data_usd['fixedFrequency']['rpm.m4']
    
    return dts, states, controls

def load_csv(file_name):
    """
    Loads (simulated) quadrotor data stored in csv file format and creates tensor dataset
    """
    data = np.loadtxt(file_name, delimiter=',')

    dts = np.diff(data[:,0])
    dt = np.mean(dts)

    data_torch = torch.from_numpy(data[:,1:])  # skip the dt column
    x = data_torch[:-1,:]

    y = data_torch[1:, :13]

    dataset = TensorDataset(x,y)
    return dt, dataset

def load_dataset(file_name):
    """
    Loads a pickled TensorDataset
    """
    dataset = torch.load(file_name)

    m = re.search('_([0-9]+)Hz',file_name)
    if m:
        hz = float(m.group(1))
        dt = 1/hz
    else:
        dt = 0.01

    return dt, dataset

def run_trajectory(model, controls, s_0):
    model.eval()
    with torch.no_grad():
        states = model(controls.unsqueeze(1), s_0=s_0)
    return states.squeeze(1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--file-train", type=str, default='figure8_100hz')
    parser.add_argument("--file-test", type=str, default='hover_100hz')
    parser.add_argument("--window-size", type=int, default=3,
                        help='the length of the time windows the trajectory is cut into for training')
    parser.add_argument("--lr", type=float, default=1.0,
                        help='the learning rate of the optimizer')
    parser.add_argument("--epochs", type=int, default=100,
                        help='number of epochs to run the optimization')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='the size of the batches for training')
    parser.add_argument('--double-window-size-on-plateau', type=lambda x: bool(strtobool(x)), default=True,
                        help='if toggled, the window size will double if the training loss does not decrease any more')
    parser.add_argument('--num-epochs-plateau', type=int, default=5,
                        help='number of epochs to wait on plateau before window size is doubled')
    parser.add_argument('--visualize-trajectory', type=lambda x: bool(strtobool(x)) ,default=True,
                        help='Toggles whether or not the trajectory should be visualized after training')
    parser.add_argument('--report-parameters', type=lambda x: bool(strtobool(x)) ,default=True,
                        help='Toggles whether or not the current gains are reported during training')
    parser.add_argument('--track', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--max-window-size', type=int, default=400,
                        help='maximum window size to extend the trajectories to')
    args = parser.parse_args()

    train_file_path = os.path.join(args.data_dir, args.file_train)
    test_file_path = os.path.join(args.data_dir, args.file_test)
    
    torch.autograd.set_detect_anomaly(True)

    train_dataset = FlightDataset(file_name=train_file_path, transform=torch.tensor)
    train_dataset.slice_into_windows(window_size=args.window_size)
    test_dataset = FlightDataset(file_name=test_file_path, transform=torch.tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    quadrotor_simulation_module = QuadrotorSimulationModule(mass=1.0, inertia=[0.01, 0.01, 0.01])

    criterion = QuadrotorLoss()
    optimizer = optim.SGD(quadrotor_simulation_module.parameters(), lr=args.lr)

    run_name = f'quadrotor_system_id__{int(time.time())}'
    if args.track:
        writer = SummaryWriter(f"runs/parameters_{run_name}")
    else:
        writer = None

    if args.double_window_size_on_plateau:
        best_loss = torch.inf
        iterations_since_decrease = 0

    for epoch in range(args.epochs):
        loss, position_loss, velocity_loss, angle_loss, omega_loss = train_quadrotor_system_identification(
            model=quadrotor_simulation_module, criterion=criterion, optimizer=optimizer, trainloader=train_dataloader
        )
        
        if writer is not None:
            writer.add_scalar("loss/position_loss", position_loss, global_step=epoch)
            writer.add_scalar("loss/velocity_loss", velocity_loss, global_step=epoch)
            writer.add_scalar("loss/angle_loss", angle_loss, global_step=epoch)
            writer.add_scalar("loss/omega_loss", omega_loss, global_step=epoch)
            writer.add_scalar("parameters/window_size", args.window_size, global_step=epoch)
            writer.add_scalar("physical_parameters/mass", quadrotor_simulation_module.mass, global_step=epoch)
            writer.add_scalar("physical_parameters/inertia", quadrotor_simulation_module.inertia, global_step=epoch)

        if args.double_window_size_on_plateau:
            if loss < best_loss - 1e-4:
                best_loss = loss
                iterations_since_decrease = 0
            else:
                iterations_since_decrease += 1
            
            if iterations_since_decrease > args.num_epochs_plateau:
                args.window_size *= 2
                if args.window_size > args.max_window_size:
                    args.window_size = args.max_window_size
                train_dataset.slice_into_windows(window_size=args.window_size)
                iterations_since_decrease = 0
                best_loss = torch.inf
        
        if args.report_parameters:
            with torch.no_grad():
                print(f"Parameters after {epoch+1} epochs:\nmass={quadrotor_simulation_module.mass}\tinertia={quadrotor_simulation_module.inertia.tolist()}")
        
    if writer is not None:
        writer.close()
    
    if args.visualize_trajectory:
        controls = torch.tensor(train_dataset.raw_controls, dtype=torch.double)
        true_states = torch.tensor(train_dataset.raw_states, dtype=torch.double)
        simulated_states = run_trajectory(quadrotor_simulation_module, controls=controls, s_0=true_states[:1,:])
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(simulated_states[:,0], simulated_states[:,1], simulated_states[:,2], label='simulated trajectory')
        ax.plot(true_states[:,0], true_states[:,1], true_states[:,2], label='true trajectory')
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=-35, roll=0)
        plt.show()


    ### 
    # Old code


    # if train_file_path.endswith('.csv'):
    #     dt, training_data = load_csv(train_file_path)
    # elif train_file_path.endswith('.pt'):
    #     dt, training_data = load_dataset(train_file_path)
    # else:
    #     dt, training_data = load_cfusd(train_file_path)
    
    # if test_file_path.endswith('.csv'):
    #     dt2, test_data = load_csv(test_file_path)
    # elif test_file_path.endswith('.pt'):
    #     dt2, test_data = load_dataset(test_file_path)
    # else:
    #     dt2, test_data = load(test_file_path)

    # train_dataloader = DataLoader(training_data, batch_size=1024, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size=1024)


    # model = QuadrotorSimulationModule(dt, mass=1.)

    # # loss_fn = nn.MSELoss()
    # loss_fn = QuadrotorLoss()

    # learning_rate = 1
    # epochs = 1000
    # train_losses, test_losses = [], []

    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # for t in range(epochs):
    #     print(f"Epoch {t+1}\n-------------------------------")
    #     train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
    #     test_loss = test_loop(test_dataloader, model, loss_fn)
    #     train_losses.append(train_loss)
    #     test_losses.append(test_loss)
    # print("Done!")
    # print(model.state_dict())

    # plt.plot(range(epochs), train_losses, label="training error")
    # plt.plot(range(epochs), test_losses, label="test error")
    # plt.xlabel('epoch')
    # plt.ylabel('error')
    # plt.legend()
    # plt.title("Train and test error over epochs")
    # plt.show()



