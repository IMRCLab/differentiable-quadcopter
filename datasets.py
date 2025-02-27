import numpy as np
import os

from data import cfusdlog
from torch.utils.data import Dataset
from utils import slice_dataset

def load_cfusd(filename, skip_first=0, drop_last=0):
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
    data_usd = cfusdlog.decode(filename)

    T = len(data_usd['fixedFrequency']['timestamp']) - skip_first - drop_last
    
    if drop_last == 0:
        drop_last = None
    else:
        drop_last = -drop_last

    dts = np.diff(data_usd['fixedFrequency']['timestamp'][skip_first:drop_last]) / 1000.0
    # dt = np.mean(dts)


    states = np.empty((T, 13), dtype=np.float64)
    controls = np.empty((T,4), dtype=np.float64)

    if filename.endswith('hz'):
        states[:, 0] = data_usd['fixedFrequency']['stateEstimateZ.x'][skip_first:drop_last] / 1000.0
        states[:, 1] = data_usd['fixedFrequency']['stateEstimateZ.y'][skip_first:drop_last] / 1000.0
        states[:, 2] = data_usd['fixedFrequency']['stateEstimateZ.z'][skip_first:drop_last] / 1000.0
        states[:, 3] = data_usd['fixedFrequency']['stateEstimateZ.vx'][skip_first:drop_last] / 1000.0
        states[:, 4] = data_usd['fixedFrequency']['stateEstimateZ.vy'][skip_first:drop_last] / 1000.0
        states[:, 5] = data_usd['fixedFrequency']['stateEstimateZ.vz'][skip_first:drop_last] / 1000.0
        states[:, 10] = data_usd['fixedFrequency']['stateEstimateZ.rateRoll'][skip_first:drop_last] / 1000.0
        states[:, 11] = data_usd['fixedFrequency']['stateEstimateZ.ratePitch'][skip_first:drop_last] / 1000.0
        states[:, 12] = data_usd['fixedFrequency']['stateEstimateZ.rateYaw'][skip_first:drop_last] / 1000.0
        for t in range(T):
            # q is in [x,y,z,w] format
            q = cfusdlog.quatdecompress(
                data_usd['fixedFrequency']['stateEstimateZ.quat'][t+skip_first])
            # [w,x,y,z] format
            states[t, 6:7] = q[3:4]
            states[t, 7:10] = q[0:3]
    else:
        states[:, 0] = data_usd['fixedFrequency']['stateEstimate.x'][skip_first:drop_last] / 1000.0
        states[:, 1] = data_usd['fixedFrequency']['stateEstimate.y'][skip_first:drop_last] / 1000.0
        states[:, 2] = data_usd['fixedFrequency']['stateEstimate.z'][skip_first:drop_last] / 1000.0
        states[:, 3] = data_usd['fixedFrequency']['stateEstimate.vx'][skip_first:drop_last] / 1000.0
        states[:, 4] = data_usd['fixedFrequency']['stateEstimate.vy'][skip_first:drop_last] / 1000.0
        states[:, 5] = data_usd['fixedFrequency']['stateEstimate.vz'][skip_first:drop_last] / 1000.0
        states[:, 6] = data_usd['fixedFrequency']['stateEstimate.qw'][skip_first:drop_last]
        states[:, 7] = data_usd['fixedFrequency']['stateEstimate.qx'][skip_first:drop_last]
        states[:, 8] = data_usd['fixedFrequency']['stateEstimate.qy'][skip_first:drop_last]
        states[:, 9] = data_usd['fixedFrequency']['stateEstimate.qz'][skip_first:drop_last]
        states[:, 10] = data_usd['fixedFrequency']['gyro.x'][skip_first:drop_last] / 1000.0
        states[:, 11] = data_usd['fixedFrequency']['gyro.y'][skip_first:drop_last] / 1000.0
        states[:, 12] = data_usd['fixedFrequency']['gyro.z'][skip_first:drop_last] / 1000.0

    if filename.startswith('nn_'):
        controls[:, 0] = data_usd['fixedFrequency']['rpm.m1'][skip_first:drop_last]
        controls[:, 1] = data_usd['fixedFrequency']['rpm.m2'][skip_first:drop_last]
        controls[:, 2] = data_usd['fixedFrequency']['rpm.m3'][skip_first:drop_last]
        controls[:, 3] = data_usd['fixedFrequency']['rpm.m4'][skip_first:drop_last]
    else:
        controls[:, 0] = data_usd['fixedFrequency']['rpm.m1'][skip_first:drop_last]
        controls[:, 1] = data_usd['fixedFrequency']['rpm.m2'][skip_first:drop_last]
        controls[:, 2] = data_usd['fixedFrequency']['rpm.m3'][skip_first:drop_last]
        controls[:, 3] = data_usd['fixedFrequency']['rpm.m4'][skip_first:drop_last]
    
    return dts, states, controls


class BaseTrajectoryDataset(Dataset):
    """
    Base class for trajectory datasets. Provides interface for loading and slicing of raw data into windows.
    """
    def __init__(self, data_path, transform, window_size=None):
        """
        Parameters:
        -----------
        data_path: str
            Path to the data file or directory with data files
        transform: callable
            Transformation to be applied to the data
        window_size: int
            Size of the window to slice the data into
        """
        super().__init__()
        self._load_raw_data(data_path)
        self.window_size = window_size
        self.transform = transform
        if self.window_size is None:
            self.window_size = min([len(raw_states) for raw_states in self.raw_states])
        self._slice_into_windows(window_size=self.window_size)
    
    def _load_raw_data(self, data_path):
        raise NotImplementedError()
    
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

    def _slice_into_windows(self, window_size):
        """
        Slices the raw data into windows of a given length.

        Parameters:
        -----------
        window_size: int
            Size of the window to slice the data into
        """
        max_window_size = min([len(raw_states) for raw_states in self.raw_states])
        if window_size > max_window_size:
            window_size = max_window_size
        self.window_size = window_size
        self.states = [slice_dataset(raw_states, window_size=window_size) for raw_states in self.raw_states]
        self.states = np.concat(self.states, axis=0)

        dts_splits = [] 
        control_splits = []
        for raw_dts, raw_controls in zip(self.raw_dts, self.raw_controls):
            dataset_size = len(raw_dts)
            N = dataset_size // (window_size-1)  # number of segments
            dts_splits += np.split(raw_dts[:N*(window_size-1)], N, axis=0)
            control_splits += np.split(raw_controls[:N*(window_size-1)], N, axis=0)
        dts = np.stack(dts_splits, axis=0)
        self.dts = np.expand_dims(dts, axis=-1)
        self.controls = np.stack(control_splits, axis=0)


class FlightDataset(BaseTrajectoryDataset):
    """
    Dataset class for the Crazyflie flight data.
    """
    def __init__(self, data_path, transform, window_size=None):
        super().__init__(data_path, transform, window_size)

    def _load_raw_data(self, path):
        """
        Load the raw data from the given path. The path can be either a file or a directory.
        
        Parameters:
        ------------
        path: str
            Path to the data file or directory with data
        """
        if os.path.isfile(path):
            raw_dts, raw_states, raw_controls = load_cfusd(path)
            self.raw_dts, self.raw_states, self.raw_controls = [raw_dts], [raw_states], [raw_controls]
        elif os.path.isdir(path):
            self.raw_dts, self.raw_states, self.raw_controls = [], [], []
            for file_name in os.listdir(path):
                file_path = os.path.join(path, file_name)
                raw_dts, raw_states, raw_controls = load_cfusd(file_path)
                self.raw_dts.append(raw_dts)
                self.raw_states.append(raw_states)
                self.raw_controls.append(raw_controls)
        else:
            raise ValueError(f'{path} is neither file nor directory.')
        

class SimulatedDataset(BaseTrajectoryDataset):
    """
    Dataset class for the simulated data.
    """
    def __init__(self, data_path, transform, window_size=None):
        super().__init__(data_path, transform, window_size)
    
    def _load_raw_data(self, path):
        """
        Load the raw data from the given path. The path can be either a file or a directory.

        Parameters:
        ------------
        path: str
            Path to the data file or directory with data
        """
        if os.path.isfile(path):
            data = np.load(path)
            self.raw_dts = [np.diff(data[:,0])]
            self.raw_states = [data[:,1:14]]
            self.raw_controls = [data[:, 14:18]]
        elif os.path.isdir(path):
            self.raw_dts, self.raw_states, self.raw_controls = [], [], []
            for file_name in os.listdir(path):
                file_path = os.path.join(path, file_name)
                data = np.load(file_path)
                raw_dts = np.diff(data[:,0])
                raw_states = data[:,1:14]
                raw_controls = data[:, 14:18]
                self.raw_dts.append(raw_dts)
                self.raw_states.append(raw_states)
                self.raw_controls.append(raw_controls)
        else:
            raise ValueError(f'{path} is neither file nor directory.')
