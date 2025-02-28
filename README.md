# differentiable-quadcopter

## Generating data
The utilities for the trajectories are collected in `trajectories.py`. To show example plot for positions, velocities and higher order derivatives run:
```
python trajectories.py
```

## Tensorboard
The default location for the logs is the `runs` directory:
```bash
tensorboard --logdir runs
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

## System identification
The script `train_system_id.py` contains code to run system identification for a quadrotor. To run the system identification you need flight data with states and controls. These can be either simulated (which is indicated by the `--control-type forces` flag) or recorded real flight data (which is indicated by the `--control-type rpm` flag).
An example can be run with following command (from the root of the project):

```bash
python train_system_id.py --data-dir . --file-train data/system_id/achim/achim_16 --control-type rpm --track --report-parameters n --epochs 100 --window-size 2 --max-window-size 2 --clip-gradient-norm 0.5 --batch-size 10 --skip-first-n 500 --drop-last-n 2500 --lr 0.1 --run-name 'achim_16_fim_batch_size_10_cov_batch' --use-fim True --compute-batch-fim True
```

### Natural gradient using approximate Fisher information matrix
In system identification we face the problem that for some state-input combinations not all parameters are not identifiable e.g. the inertia around the x-axis is not identifiable for rotations which are only around the y-axis. Gradients w.r.t. these unindentifiable parameters are therefore meaningless and should not be used for system identification. To account for the identifiability during the optimization procedure we can use the natural gradient where we project the gradient for each sample using the sensitivity based Fisher information matrix.
A gradient step becomes:
$$
\theta \leftarrow \theta + F_i^{-1} \frac{\partial L_i}{\partial \theta}
$$
Where $L_i$ is the loss for the $i$-th sample and $F_i$ is the Fisher information matrix for the $i$-th sample:
$$
F_i = \left(\frac{\partial L_i}{\partial \theta} \right) C \left( \frac{\partial L_i}{\partial \theta}\right)^T
$$

The vector $\theta$ are the stacked parameters of the UAV $\theta = (I_{xx}, I_{yy}, I_{zz}, m)^T$. And $C$ is the empirical covariance matrix:
$$
C =  \frac{1}{N} \sum_{i=0}^{N}\left(\frac{\partial L_i}{\partial \theta} - \bar{g}\right)\left(\frac{\partial L_i}{\partial \theta} - \bar{g}\right)^T
$$
Here $\bar{g}$ denotes the mean gradient over the batch.

The system identification script then provides two modes - in the `use_batch_fim = True` the Fisher information matrix is computed as the expectation over the sample FIMs and for each sample the same empirical mean FIM is used.
$$
F = \frac{1}{N} \sum_{i=0}^N F_i
$$
In the `use_batch_fim = False` the individual FIMs are kept and each gradient is projected with its own FIM estimate.

### Further investigation
The conditioning of the FIM is extremly poor and the smallest eigenvalues are sometimes around `10e-32`. Currently we consider only FIMs with a smallest eigenvalue $\lambda_{min}$ as too bad conditioned - this might be too generous. However for small inertia values the FIM becomes even worse conditioned during convergence.
One way to approach this problem could be to use a singular-value decomposition (SVD) of the FIM and filter out too uninformative directions leading to a low rank approximation of the FIM. 
