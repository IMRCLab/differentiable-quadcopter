from datasets import FlightDataset, SimulatedDataset
import roma
from pathlib import Path

import torch
import utils_visualize as vis

from quadrotor_pytorch import QuadrotorAutograd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file-name', type=str, default='simulated_trajectories/simulated_excitatory_reg_traj_stretch_1_0.npy')
parser.add_argument('--control-type', type=str, default='forces')
parser.add_argument('--skip-first-n', type=int, default=0)

args = parser.parse_args()

# print(dataset.raw_states.shape, dataset.raw_dts.shape)
# print(dataset.raw_dts)
datadir = Path().absolute().joinpath('data')
if args.file_name.endswith('.npy'):
    dataset = SimulatedDataset(datadir.joinpath(args.file_name), transform=torch.Tensor)
else:
    dataset = FlightDataset(datadir.joinpath(args.file_name).as_posix(), transform=torch.tensor)
# dataset = FlightDataset(datadir.joinpath('system_id/achim/achim_14').as_posix(), transform=torch.tensor)
control_type = args.control_type

# dataset = SimulatedDataset(datadir.joinpath(args.file_name), transform=torch.Tensor)

dataset.raw_states = [raw_states[args.skip_first_n:,...] for raw_states in dataset.raw_states]
dataset.raw_controls = [raw_controls[args.skip_first_n:,...] for raw_controls in dataset.raw_controls]
dataset.raw_dts = [raw_dts[args.skip_first_n:,...] for raw_dts in dataset.raw_dts]



true_states = dataset.raw_states[0]
controls = dataset.raw_controls[0]

quadrotor = QuadrotorAutograd()

vis.plot_trajectory(true_states, title=f'trajectory: {args.file_name}')
if control_type=='rpm':
    controls = quadrotor._rpm_to_force(controls=torch.Tensor(controls))
vis.plot_controls(controls, min_u=quadrotor.min_u, max_u=quadrotor.max_u)
