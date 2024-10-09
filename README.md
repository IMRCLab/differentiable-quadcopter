# differentiable-quadcopter

## Generating data
The utilities for the trajectories are collected in `trajectories.py`. To show example plot for positions, velocities and higher order derivatives run:
```
python trajectories.py
```

## The UAV model
The descripition of the UAV model with its physics model is contained in `quadrotor_pytorch.py`. The class `QuadrotorAutograd` implements the dynamics for a quadrotor with a given mass $m$ and inertia matrix $I$. Given some control forces the state is propagated using the `step` function. All functions are implemented using PyTorch allowing for automatic differentiation of the dynamics.

## The controller
The controller is implemented in the file `controller_pytorch.py`. The class `ControllerLee` implements the function `compute_controls` which computes the total thrust, the torque vecotor and additionally the desired attitude and rotational velocity given the current state and a corresponding setpoint.

## Training controller and quadrotor
The controller and the UAV model together allow for simulation and automatic tuning of the controllers gains. This training procedure is implemented in the `train_lee_controller.py` file. The class `QuadrotorControllerModule` is a `torch.nn.Module` subclass and registers the gains as trainable torch parameters. The `forward` takes a trajectory of desired setpoints as input and simulates the UAV model for the length of this trajectory. The output is then the simulated trajectory.

The file also contains a custom dataset class `NthOrderTrajectoryDataset` which generates a dataset of setpoints from a list of splines (and the derivates up to n-th order), coefficients parameterizing the splines and an array of timepoints at which the splines should be evaluated. The function `slice_into_windows` slices a single trajectory into successive windows of a given length.

An example of the dataset creation and the simulation loop can be run with:
```
python train_lee_controller.py --epochs 5000 --lr 1e-3 --track
```

The script `train_lee_controller.py` supports different CLI arguments. For a detailed list have a look at the help menu:
```
python train_lee_controller.py -h
```

## Evaluation of the controller in simulation
The script `evaluate_lee_controller.py` contains the necessary code to evaluate fitted gains against some baseline gains. The evaluation runs on a couple of trajectories. So far a helix, a circle, the figure 8 and and random waypoints are included. By default the results include the parameters and the losses for the models on different trajectory. The results are written to the `results.yaml` file and stored in the folder `results`. Figures of the trajectories are saved in the folder `figures` within the `results` folder.

To run the evaluation simply execute the evaluation script:
```
python evaluate_lee_controller.py
```

TODO: Add command line arguments to the evaluation script