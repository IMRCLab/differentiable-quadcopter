import argparse

import numpy as np
import torch
from torch.utils.data import TensorDataset

from quadrotor_pytorch import QuadrotorAutograd
from train_lee_controller import QuadrotorControllerModule, NthOrderTrajectoryDataset

from trajectories import f, fdot, fdotdot, fdotdotdot

import utils_visualize as vis

class QuadrotorControllerSimulationModule(QuadrotorControllerModule):
    """
    Basically the same as the QuadrotorControllerModule but with explicit support of passing 
    seperate inertia and mass to the quadrotor and the controller.
    """
    def __init__(self, dt, kp=[[1.],[1.],[1.]], kv=[[1.],[1.],[1.]], kw=[[1.],[1.],[1.]], kr=[[1.],[1.],[1.]], controller_mass=None, controller_inertia=None, noise_on=False, collect_controls=False, quadrotor_mass=None, quadrotor_inertia=None):
        super().__init__(dt, kp=kp, kv=kv, kw=kw, kr=kr, mass=controller_mass, inertia=controller_inertia, noise_on=noise_on, collect_controls=collect_controls)
        if quadrotor_mass is not None:
            self.quadrotor.m = torch.tensor(quadrotor_mass, dtype=torch.double)
        if quadrotor_inertia is not None:
            self.quadrotor.I = torch.tensor(quadrotor_inertia, dtype=torch.double)


class QuadrotorSimulator():
    """
    Class to simulate the quadrotor dynamics
    """
    def __init__(self, dt, mass=None, J=None):
        super(QuadrotorSimulator, self).__init__()
        self.dynamics_model = QuadrotorAutograd()
        self.dynamics_model.dt = dt
        if mass is not None:
            self.dynamics_model.mass = mass
        if J is not None:
            self.dynamics_model.J = J # inertia

        self.kf = 2.1
    
    def forward(self, state, action):
        """
        Forward propagate the state using the controls and the dynamics.
        Returns the next state.

        Parameters:
        -----------
            x: torch.Tensor
                Tensor of state and action
        
        Returns:
        --------
            next_state: torch.Tensor
                Tensor with the next state
        """
        force = self.kf * 1e-10 * torch.pow(action, 2)
        next_state = self.dynamics_model.step(state, force)
        return next_state

    def generate_trajectory(self, T, file=None):
        """
        Generate a trajectory with a given length

        Parameters:
        -----------
            T: int
                length of the trajectory
            
            file: str
                If given - file to write to. Default None
        """
        # data with columns: timestamp, x, y, z, vx, vy, vz, qx, qy, qz, qw, roll, pitch, yaw, m1, m2, m3, m4
        data = torch.empty((T, 18),dtype=torch.float64)

        # set initial state
        state = torch.tensor([[0.,0.,0.5, 0.,0.,0.,1.,0.,0.,0.,0.,0.,0.]],dtype=torch.float64)
        action = torch.rand((1,4), dtype=torch.float64)
        action = action * (25_000 - 21_000) + 21_000
        data[0,0] = 0.0 # starting time
        data[0:1,1:14] = state
        data[0:1,14:18] = action
        for t in range(1,T):
            # propagate state
            state = self.forward(state, action)
            # sample new action
            action = torch.rand((1,4), dtype=torch.float64)
            action = action * (25_000 - 21_000) + 21_000
            # safe data
            data[t,0] = t*self.dynamics_model.dt
            data[t:t+1,1:14] = state
            data[t:t+1,14:18] = action

        if file:
            np.savetxt(file, data.numpy(), delimiter=',')
        
        return data
    
    def generate_pairwise_dataset(self, N, file=None):
        x = torch.empty((N,17), dtype=torch.float64)
        y = torch.empty((N,13), dtype=torch.float64)

        for i in range(N):
            state = torch.tensor([[0.,0.,0.5, 0.,0.,0.,1.,0.,0.,0.,0.,0.,0.]],dtype=torch.float64)
            state += torch.randn_like(state) 
            action = torch.rand((1,4), dtype=torch.float64)
            action = action * (25_000 - 21_000) + 21_000
            next_state = self.forward(state, action)
            x[i,:13] = state
            x[i,13:] = action
            y[i,:] = next_state
        
        dataset = TensorDataset(x, y)
        if file is not None:
            torch.save(dataset, file)

def simulate_trajectory_with_controls(trajectory_path, dt, visualize=True):
    """
    Simulate a trajectory with a given path
    
    Parameters:
    -----------
        trajectory_path: str
            Path to the trajectory file
        
        dt: float
            Sampling rate for the trajectory
        
        visualize: bool
            If True, visualize the trajectory
    """
    # load dataset
    trajectory_data = NthOrderTrajectoryDataset(trajectory_path,[f, fdot, fdotdot, fdotdotdot], dt=dt, transform=torch.tensor)

    # create simulation module
    simulation = QuadrotorControllerSimulationModule(dt=dt,kp=9.0, kv=7.0, kr=0.0055, kw=0.0013, collect_controls=True)

    # simulate trajectory
    states, _, _, controls = simulation(trajectory_data[0][:,None,:])

    states = states.detach().numpy()[:,0,:]
    controls = controls.detach().numpy()[:,:,0]
    t = args.dt * np.arange(0, controls.shape[0])[:,None]
    x = np.concat([t, states, controls], axis=1)

    # save simulation data
    trajectory_name = trajectory_path.split('.')[0]
    np.save(f'simulated_{trajectory_name}.npy', x)

    # visualize trajectory
    if visualize:
        # plot the trajectories (desired vs true)
        vis.plot_trajectory(states, trajectory_data[0])


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-name", type=str, default='figure8.csv', help='file name to save data')
    parser.add_argument("--N", type=int, default=1000, help='trajectory length to generate')
    parser.add_argument("--dt", type=float, default=0.01, help='sampling rate for trajectory')
    parser.add_argument("--generate-pairwise", type=bool, default=False)
    args = parser.parse_args()

    simulate_trajectory_with_controls(args.file_name, args.dt, visualize=True)