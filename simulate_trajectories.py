import argparse

import numpy as np
import torch

from quadrotor_pytorch import QuadrotorAutograd

class QuadrotorSimulator():
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

        # sample random starting positions
        state = torch.rand((1,13), dtype=torch.float64)
        state = state * (self.dynamics_model.max_x - self.dynamics_model.min_x) + self.dynamics_model.min_x
        action = torch.rand((1,4), dtype=torch.float64)
        action = action * (25_000 - 0) + 0
        data[0,0] = 0.0 # starting time
        data[0:1,1:14] = state
        data[0:1,14:18] = action
        for t in range(1,T):
            # propagate state
            state = self.forward(state, action)
            # sample new action
            action = torch.rand((1,4), dtype=torch.float64)
            action = action * (25_000 - 0) + 0
            # safe data
            data[t,0] = t*self.dynamics_model.dt
            data[t:t+1,1:14] = state
            data[t:t+1,14:18] = action

        if file:
            np.savetxt(file, data.numpy(), delimiter=',')
        
        return data


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-name", type=str, help='file name to save data')
    parser.add_argument("--T", type=int, default=1000, help='trajectory length to generate')
    args = parser.parse_args()

    simulator = QuadrotorSimulator(dt=0.01)

    simulator.generate_trajectory(T=args.T, file=args.file_name)