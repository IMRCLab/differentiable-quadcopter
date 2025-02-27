import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import time
import torch
from tqdm import tqdm



from distutils.util import strtobool
from torch.utils.data.dataset import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from quadrotor_pytorch import QuadrotorAutograd 

from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from datasets import FlightDataset, SimulatedDataset
from losses import QuadrotorLoss

import utils_visualize as vis


class QuadrotorSimulationModule(nn.Module):
    def __init__(self, control_type='rpm', **kwargs):
        super().__init__()
        self.quadrotor = QuadrotorAutograd(**kwargs)
        self.control_type = control_type

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
        if self.control_type == 'rpm':
            controls = self.quadrotor._rpm_to_force(controls)
        elif self.control_type == 'pwm':
            raise NotImplementedError()
        elif self.control_type != 'forces':
            raise ValueError(f"Unknown control type {self.control_type}")

        for i, control in enumerate(controls):
            if dts is not None:
                dt = dts[i]
            else:
                dt = self.quadrotor.dt
            current_state = self.quadrotor.step(state=current_state, force=control, dt=dt)
            states += [current_state]
        return torch.stack(states, dim=0)
    
def compute_sample_grad(model, s_0, controls, dts, targets):
    """
    Compute the gradient of the loss w.r.t. the model parameters for a single sample.
    """
    s_0 = s_0.unsqueeze(0)
    controls = controls.unsqueeze(0)
    dts = dts.unsqueeze(0)
    targets = targets.unsqueeze(0)

    # compute simulated next states
    simulated_states = model(s_0=s_0, controls=controls.transpose(0,1), dts=dts.transpose(0,1))
    position_loss, velocity_loss, angle_loss, omega_loss = criterion(simulated_states.transpose(0,1)[:,1:,:], targets)
    loss = position_loss + velocity_loss + angle_loss + omega_loss
    grad = torch.autograd.grad(loss, model.parameters(), create_graph=False)
    return grad

def compute_grad(model, s_0, controls, dts, targets):
    batch_size = s_0.shape[0]
    sample_grads = [compute_sample_grad(model, s_0[i], controls[i], dts[i], targets[i]) for i in range(batch_size)]
    sample_grads = zip(*sample_grads)
    sample_grads = [torch.stack(shards) for shards in sample_grads]
    return sample_grads

def compute_sample_jacobian(model, s_0, controls, dts):
    """
    Compute the jacobian of the simulated states w.r.t. the model parameters for a single sample.
    """
    s_0 = s_0.unsqueeze(0)
    controls = controls.unsqueeze(0)
    dts = dts.unsqueeze(0)

    # compute simulated next states
    simulated_states = model(s_0=s_0, controls=controls.transpose(0,1), dts=dts.transpose(0,1))
    simulated_targets = simulated_states.transpose(0,1)[:,1:,:] # skip the first state

    grads = []

    # compute the gradient for each element in the state vector
    for i in range(simulated_targets.shape[-1]):
        grad = torch.autograd.grad(simulated_targets[..., i], model.parameters(), create_graph=True)
        grads.append(grad)
    grads = zip(*grads)
    jacobians = [torch.stack(shards) for shards in grads]
    return jacobians

def compute_sample_fim(model, s_0, controls, dts):
    jacobian = compute_sample_jacobian(model, s_0, controls, dts)
    jacobian = torch.cat(jacobian, axis=-1)
    return jacobian.T @ jacobian

def compute_fim(model, s_0, controls, dts):
    batch_size = s_0.shape[0]
    sample_fims = [compute_sample_fim(model, s_0[i], controls[i], dts[i]) for i in range(batch_size)]
    fims = torch.stack(sample_fims)
    return fims

# TODO: adapt the code to work for the stochastic case e.g. real world flight data with noise and delays

def train_quadrotor_system_identification(model, criterion, optimizer, trainloader, clip_gradient_norm=0.1, use_fim=True, compute_batch_fim=True):
    """
    Train the quadrotor system identification model for one epoch.
    
    Parameters:
    -----------
    model: nn.Module
        the model to be trained
    criterion: nn.Module
        the loss function to be optimized
    optimizer: torch.optim.Optimizer
        the optimizer to be used for optimization
    trainloader: DataLoader
        the dataloader providing the training data
    clip_gradient_norm: float
        the maximum norm of the gradients relative to the parameter values
    use_fim: bool
        if toggled, the Fisher Information Matrix will be used for optimization
    compute_batch_fim: bool
        if toggled, the Fisher Information Matrix will be computed as average over each batch
    """
    running_loss, running_position_loss, running_velocity_loss, running_angle_loss, running_omega_loss = 0.0, 0.0, 0.0, 0.0, 0.0

    model.train()
    for batch_idx, (recorded_dts, recorded_states, recorded_controls) in enumerate(trainloader):  # recorded states
        optimizer.zero_grad()
        if use_fim:
            batch_grads = compute_grad(model, recorded_states[:,0,:], recorded_controls, recorded_dts, recorded_states[:,1:,:])
            # compute the approximate Fisher Information Matrix
            batch_fims = compute_fim(model, recorded_states[:,0,:], recorded_controls, recorded_dts)
            with torch.no_grad():
                # compute the projected gradients using the FIM
                grads = torch.concat(batch_grads, dim=-1).unsqueeze(2)
                if compute_batch_fim:
                    batch_fims = batch_fims.mean(axis=0,keepdim=True)
                    grads = grads.mean(axis=0,keepdim=True)
                scaled_grads = torch.linalg.solve(batch_fims, grads).squeeze(2)

                mass_grads = scaled_grads[:,:1]
                inertia_grads = scaled_grads[:,1:]
                batch_grads = [mass_grads, inertia_grads]

                for parameter, grad in zip(model.parameters(), batch_grads):
                    parameter.grad = grad.mean(axis=0)

                simulated_states = model(s_0=recorded_states[:,0,:], controls=recorded_controls.transpose(0,1), dts=recorded_dts.transpose(0,1))
                position_loss, velocity_loss, angle_loss, omega_loss = criterion(simulated_states.transpose(0,1)[:,1:,:], recorded_states[:,1:,:])
                loss = position_loss + velocity_loss + angle_loss + omega_loss

        if not use_fim:
            simulated_states = model(s_0=recorded_states[:,0,:], controls=recorded_controls.transpose(0,1), dts=recorded_dts.transpose(0,1))
            position_loss, velocity_loss, angle_loss, omega_loss = criterion(simulated_states.transpose(0,1)[:,1:,:], recorded_states[:,1:,:])
            loss = position_loss + velocity_loss + angle_loss + omega_loss
            loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                with torch.no_grad():
                    # gradient projection
                    relative_grad = parameter.grad / parameter
                    r_max = 0.0
                    for r in relative_grad:
                        if torch.abs(r) > clip_gradient_norm and torch.abs(r) > r_max:
                            r_max = torch.abs(r)
                    if r_max > 0.0:
                        parameter.grad = parameter.grad / r_max * clip_gradient_norm
        optimizer.step()

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.clamp_(min=1e-8)
        
        running_loss += loss.item()
        running_position_loss += position_loss.item()
        running_velocity_loss += velocity_loss.item()
        running_angle_loss += angle_loss.item()
        running_omega_loss += omega_loss.item()
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
    parser.add_argument("--file-train", type=str, default='figure8_500hz')
    parser.add_argument("--file-test", type=str, default='hover_500hz')
    parser.add_argument("--window-size", type=int, default=2,
                        help='the length of the time windows the trajectory is cut into for training')
    parser.add_argument("--lr", type=float, default=0.1,
                        help='the learning rate of the optimizer')
    parser.add_argument("--epochs", type=int, default=1000,
                        help='number of epochs to run the optimization')
    parser.add_argument('--batch-size', type=int, default=4096,
                        help='the size of the batches for training')
    parser.add_argument('--double-window-size-on-plateau', type=lambda x: bool(strtobool(x)), default=True,
                        help='if toggled, the window size will double if the training loss does not decrease any more')
    parser.add_argument('--num-epochs-plateau', type=int, default=5,
                        help='number of epochs to wait on plateau before window size is doubled')
    parser.add_argument('--clip-gradient-norm', type=float, default=0.2,
                        help='maximum norm of the gradients relative to the parameter values')
    parser.add_argument('--visualize-trajectory', type=lambda x: bool(strtobool(x)) ,default=True,
                        help='Toggles whether or not the trajectory should be visualized after training')
    parser.add_argument('--report-parameters', type=lambda x: bool(strtobool(x)) ,default=True,
                        help='Toggles whether or not the current gains are reported during training')
    parser.add_argument('--track', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--max-window-size', type=int, default=2,
                        help='maximum window size to extend the trajectories to')
    parser.add_argument('--control-type', type=str, default='rpm',
                        help="type of the provided controls in the dataset available are 'rpm' and 'forces'")
    parser.add_argument('--skip-first-n', type=int, default=0,
                        help='number of initial states to skip in the dataset')
    parser.add_argument('--drop-last-n', type=int, default=0,
                        help='number of last states to skip in the dataset')
    parser.add_argument('--run-name', type=str, default=None,
                        help='name of the run for the tensorboard logs')
    parser.add_argument('--use-fim', type=lambda x: bool(strtobool(x)), default=True,
                        help='if toggled, the Fisher Information Matrix will be used for optimization')
    parser.add_argument('--compute-batch-fim', type=lambda x: bool(strtobool(x)), default=True,
                        help='if toggled, the Fisher Information Matrix will be computed as average over each batch')
    args = parser.parse_args()

    train_file_path = os.path.join(args.data_dir, args.file_train)
    test_file_path = os.path.join(args.data_dir, args.file_test)
    
    torch.autograd.set_detect_anomaly(True)

    if args.control_type == 'rpm':
        train_dataset = FlightDataset(data_path=train_file_path, transform=torch.tensor, window_size=args.window_size)
        test_dataset = FlightDataset(data_path=test_file_path, transform=torch.tensor)
        train_dataset.raw_states = [raw_states[args.skip_first_n:-1-args.drop_last_n,...] for raw_states in train_dataset.raw_states]
        train_dataset.raw_controls = [raw_controls[args.skip_first_n:-1-args.drop_last_n,...] for raw_controls in train_dataset.raw_controls]
        train_dataset.raw_dts = [raw_dts[args.skip_first_n:-1-args.drop_last_n,...] for raw_dts in train_dataset.raw_dts]
        train_dataset._slice_into_windows(window_size=args.window_size)
    elif args.control_type == 'pwm':
        raise NotImplementedError()
    elif args.control_type == 'forces':
        train_dataset = SimulatedDataset(data_path=train_file_path, transform=torch.tensor, window_size=args.window_size)
        test_dataset = SimulatedDataset(data_path=test_file_path, transform=torch.tensor)
        train_dataset.raw_states = [raw_states[args.skip_first_n:-1-args.drop_last_n,...] for raw_states in train_dataset.raw_states]
        train_dataset.raw_controls = [raw_controls[args.skip_first_n:-1-args.drop_last_n,...] for raw_controls in train_dataset.raw_controls]
        train_dataset.raw_dts = [raw_dts[args.skip_first_n:-1-args.drop_last_n,...] for raw_dts in train_dataset.raw_dts]
        train_dataset._slice_into_windows(window_size=args.window_size)
    else:
        raise NotImplementedError()

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    # quadrotor_simulation_module = QuadrotorSimulationModule(mass=0.1, inertia=[1e-7, 1e-7, 1e-7])
    # quadrotor_simulation_module = QuadrotorSimulationModule(mass=0.0372, inertia=[0.00040461156978587395, 0.015359155239051424, 0.012904881083249622])
    # quadrotor_simulation_module = QuadrotorSimulationModule(control_type=args.control_type, mass=0.034, inertia=[16.571710e-6, 16.655602e-6, 29.261652e-6])
    quadrotor_simulation_module = QuadrotorSimulationModule(control_type=args.control_type, mass=0.1, inertia=[0.01,0.01,0.01])

    criterion = QuadrotorLoss(reduce='mean')
    optimizer = optim.SGD(quadrotor_simulation_module.parameters(), lr=args.lr)
    
    if args.run_name is None:
        run_name = f'quadrotor_system_id__{int(time.time())}'
    else:
        run_name = args.run_name

    if args.track:
        writer = SummaryWriter(f"runs/parameters_{run_name}")
    else:
        writer = None

    if args.double_window_size_on_plateau:
        best_loss = torch.inf
        iterations_since_decrease = 0
    
    if writer is not None:
        fig, ax = vis.plot_trajectory(train_dataset.raw_states[0], title='Training trajectory', show=False)
        writer.add_figure("trajectory/trajectory", fig, global_step=0)
        writer.add_scalar("parameters/window_size", args.window_size, global_step=0)
        writer.add_scalar("physical_parameters/mass", quadrotor_simulation_module.mass, global_step=0)
        writer.add_scalars("physical_parameters/inertia", {'x':quadrotor_simulation_module.inertia[0],
                                                           'y':quadrotor_simulation_module.inertia[1],
                                                           'z':quadrotor_simulation_module.inertia[2]}, global_step=0)

    pbr = tqdm(total=args.epochs)
    for epoch in range(1,args.epochs+1):
        loss, position_loss, velocity_loss, angle_loss, omega_loss = train_quadrotor_system_identification(
            model=quadrotor_simulation_module, criterion=criterion, optimizer=optimizer, trainloader=train_dataloader, clip_gradient_norm=args.clip_gradient_norm,
            use_fim=args.use_fim, compute_batch_fim=args.compute_batch_fim
        )
        
        if writer is not None:
            writer.add_scalar("loss/position_loss", position_loss, global_step=epoch)
            writer.add_scalar("loss/velocity_loss", velocity_loss, global_step=epoch)
            writer.add_scalar("loss/angle_loss", angle_loss, global_step=epoch)
            writer.add_scalar("loss/omega_loss", omega_loss, global_step=epoch)
            writer.add_scalar("parameters/window_size", args.window_size, global_step=epoch)
            writer.add_scalar("physical_parameters/mass", quadrotor_simulation_module.mass, global_step=epoch)
            writer.add_scalars("physical_parameters/inertia", {'x':quadrotor_simulation_module.inertia[0],
                                                               'y':quadrotor_simulation_module.inertia[1],
                                                               'z':quadrotor_simulation_module.inertia[2]}, global_step=epoch)

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
                train_dataset._slice_into_windows(window_size=args.window_size)
                iterations_since_decrease = 0
                best_loss = torch.inf
        
        if args.report_parameters:
            with torch.no_grad():
                print(f"Parameters after {epoch} epochs:\nmass={quadrotor_simulation_module.mass.item()}\tinertia={quadrotor_simulation_module.inertia.tolist()}")
        pbr.update(1)
    pbr.close()
        
    if writer is not None:
        writer.close()
    
    if args.visualize_trajectory:
        controls = torch.tensor(train_dataset.raw_controls[0], dtype=torch.double)
        true_states = torch.tensor(train_dataset.raw_states[0], dtype=torch.double)
        print("####### Running the trajectory with optimized parameters #######")
        print("Mass: ", quadrotor_simulation_module.mass)
        print("Inertia: ", quadrotor_simulation_module.inertia)
        print("Length of the trajectory:",len(true_states))
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
