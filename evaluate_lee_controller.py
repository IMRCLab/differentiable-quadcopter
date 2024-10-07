import matplotlib.pyplot as plt
import os
import roma
import torch
import yaml

from train_lee_controller import QuadrotorControllerModule, NthOrderTrajectoryDataset, run_trajectory
from train_system_id import qsym_distance
from trajectories import f, fdot, fdotdot, fdotdotdot

def load_model():
    pass

def compute_errors(setpoints, states, desRs, desWs, error_fn='MSE'):
    if error_fn == 'MSE':
        error_fn = torch.nn.MSELoss(reduction='none')
    elif error_fn == 'L1':
        error_fn = torch.nn.L1Loss(reduction='none')
    else:
        raise ValueError(f'Error function {error_fn} is not supported')
    position_error = error_fn(states[...,0:3], setpoints[..., 0:3])
    velocity_error = error_fn(states[...,3:6], setpoints[..., 3:6])
    rotational_error = qsym_distance(roma.rotmat_to_unitquat(desRs), states[..., 6:10])
    omega_error = error_fn(states[..., 10:13], desWs)

    return position_error, velocity_error, rotational_error, omega_error

def plot_error_aggregated(errors, labels, title):
    """
    Plot errors as violin plots using Seaborn. Aggregates the information for the whole trajectory.
    """

    pass

def plot_error_trajectory():
    pass

if __name__=="__main__":
    result_dir = 'results'
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_dir+'/figures', exist_ok=True)

    # Workflow
    environment_files = ['figure8.csv', 'circle_0.csv', 'helix.csv', 'random_wp.csv']
    dt = 1/100 # 100Hz
    # 1. load model from checkpoint
    model = QuadrotorControllerModule(dt=dt)
    checkpoints = torch.load('checkpoints/lee_controller_best_run.pt', weights_only=True)
    model.load_state_dict(checkpoints['model_state_dict'])

    # 2. create baseline model
    baseline = QuadrotorControllerModule(dt=dt, kp=[[9.],[9.],[9.]], kv=[[7.],[7.],[7.]], kw=[[0.0013],[0.0013],[0.0013]], kr=[[0.0055],[0.0055],[0.0055]])

    result_dict = {
        'baseline_gains': {
            'kp': baseline.kp.squeeze().tolist(),
            'kv': baseline.kv.squeeze().tolist(),
            'kr': baseline.kr.squeeze().tolist(),
            'kw': baseline.kw.squeeze().tolist(),
        },
        'optimized_gains': {
            'kp': model.kp.squeeze().tolist(),
            'kv': model.kv.squeeze().tolist(),
            'kr': model.kr.squeeze().tolist(),
            'kw': model.kw.squeeze().tolist(),
        }
    }
    with open(f'{result_dir}/results.yaml', 'w') as file:
        yaml.safe_dump(result_dict, file)
    
    # 3. run model and baseline for trajectories
    for environment_file in environment_files:
        environment_name = environment_file.split('.')[0]
        # a. create dataset
        dataset = NthOrderTrajectoryDataset(parameter_file=environment_file, funcs=[f, fdot, fdotdot, fdotdotdot], dt=dt, transform=torch.tensor)

        # b. run model and baseline through dataset
        setpoint_trajectory = torch.tensor(dataset.to_setpoint(dataset.trajectory_data), dtype=torch.double)
        model_states, model_Rds, model_desWs = run_trajectory(model, setpoint_trajectory)
        baseline_states, baseline_Rds, baseline_desWs = run_trajectory(baseline, setpoint_trajectory)

        # computing errors and statistics for baseline
        position_errors_baseline, velocity_errors_baseline, rotational_errors_baseline, omega_errors_baseline = compute_errors(setpoint_trajectory, baseline_states, baseline_Rds, baseline_desWs, error_fn='MSE')
        position_mean_baseline = torch.mean(position_errors_baseline).item()
        position_std_baseline = torch.std(position_errors_baseline).item()
        velocity_mean_baseline = torch.mean(velocity_errors_baseline).item()
        velocity_std_baseline = torch.std(velocity_errors_baseline).item()
        rotational_mean_baseline = torch.mean(rotational_errors_baseline).item()
        rotational_std_baseline = torch.std(rotational_errors_baseline).item()
        omega_mean_baseline = torch.mean(omega_errors_baseline).item()
        omega_std_baseline = torch.std(omega_errors_baseline).item()


        # computing errors and statistics for model
        position_errors_model, velocity_errors_model, rotational_errors_model, omega_errors_model = compute_errors(setpoint_trajectory, model_states, model_Rds, model_desWs)
        position_mean_model = torch.mean(position_errors_model).item()
        position_std_model = torch.std(position_errors_model).item()
        velocity_mean_model = torch.mean(velocity_errors_model).item()
        velocity_std_model = torch.std(velocity_errors_model).item()
        rotational_mean_model = torch.mean(rotational_errors_model).item()
        rotational_std_model = torch.std(rotational_errors_model).item()
        omega_mean_model = torch.mean(omega_errors_model).item()
        omega_std_model = torch.std(omega_errors_model).item()

        # write error statistics to file
        error_dict = {
            f'{environment_name}': {
                'baseline': {
                    'position_error': {
                        'mean': position_mean_baseline,
                        'std': position_std_baseline,
                    },
                    'velocity_error': {
                        'mean': velocity_mean_baseline,
                        'std': velocity_std_baseline,
                    },
                    'rotational_error': {
                        'mean': rotational_mean_baseline,
                        'std': rotational_std_baseline,
                    },
                    'omega_error': {
                        'mean': omega_mean_baseline,
                        'std': omega_std_baseline,
                    },
                },
                'optimized': {
                    'position_error': {
                        'mean': position_mean_model,
                        'std': position_std_model,
                    },
                    'velocity_error': {
                        'mean': velocity_mean_model,
                        'std': velocity_std_model,
                    },
                    'rotational_error': {
                        'mean': rotational_mean_model,
                        'std': rotational_std_model,
                    },
                    'omega_error': {
                        'mean': omega_mean_model,
                        'std': omega_std_model,
                    },
                },
            }
        }

        with open(f'{result_dir}/results.yaml', 'a') as file:
            yaml.safe_dump(error_dict, file)

        # generate plots
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(baseline_states[:,0], baseline_states[:,1], baseline_states[:,2], label='baseline trajectory')
        ax.plot(model_states[:,0], model_states[:,1], model_states[:,2], label='model trajectory')
        ax.plot(setpoint_trajectory[:,0],setpoint_trajectory[:,1], setpoint_trajectory[:,2], linestyle='dotted', label='reference trajectory')
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=-35, roll=0)
        plt.tight_layout()
        plt.savefig(f'{result_dir}/figures/trajectory_{environment_name}.png')
