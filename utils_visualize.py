import matplotlib.pyplot as plt

# plot a 3d UAV trajectory from numpy data
def plot_trajectory(states, gt=None, projection='3d', title=None, show=True):
  """
  Plot the UAV trajectory in 3D or 2D.
  Parameters:
  ------------
  states: np.ndarray
      UAV trajectory data with shape (T, D)
  gt: np.ndarray
      Ground truth trajectory data with shape (T, D)
  projection: str
      Projection type, either '3d' or '2d'
  title: str
      Title of the plot
  show: bool
      Whether to show the plot
  """
  fig = plt.figure()
  ax = fig.add_subplot(projection=projection)
  if projection=='3d':
    dim = 3
  elif projection=='2d':
    dim = 2
  states=[states[:,i] for i in range(dim)]
  ax.plot(*states, label='recorded trajectory')
  if gt is not None:
    gt=[gt[:,i] for i in range(dim)]
    ax.plot(*gt, label='desired trajectory')
  ax.legend()
  if title is not None:
    ax.set_title(title)

  if show:
    plt.show()
  
  return fig, ax
  
# plot UAV trajectory states in stacked plots
def plot_state():
  raise NotImplementedError

# plot controls
def plot_controls(controls, min_u=None, max_u=None):
  """
  Plot the controls in a stacked plot. Each control is plotted in a separate subplot.
  Parameters:
  ------------
  controls: np.ndarray
      Control data with shape (T, C)
  min_u: float
      Minimum control value
  max_u: float
      Maximum control value
  """
  # plot the controls (four graphs)
  T, C = controls.shape
  fig, axs = plt.subplots(C,1)
  for i in range(C):
      axs[i].plot(controls[:,i])
      if min_u is not None:
        axs[i].axhline(y=min_u, color='r')
      if max_u is not None:
        axs[i].axhline(y=max_u, color='r')
      if max_u is not None and min_u is not None:
        axs[i].fill_between(range(T), min_u, max_u, alpha=0.2, color='r')
      # enable grid
      axs[i].grid()
  fig.suptitle('Controls')
  plt.show()

def plot_trajectory_data(trajectory, ts, title, labels):
    """
    Plot trajectory data in a stacked plot. Each coordinate is plotted in a separate subplot.
    Parameters:
    ------------
    trajectory: np.ndarray
        Trajectory data with shape (T, D)
    ts: np.ndarray
        Time steps with shape (T,)
    title: str
        Title of the plot
    labels: list
        List of labels for each coordinate e.g. ['x', 'y', 'z']
    """
    rows = len(labels) 
    fig, axs = plt.subplots(rows,1)
    fig.suptitle(title)
    fig.supxlabel('time [s]')
    for i, label in enumerate(labels):
        axs[i].plot(ts,trajectory[:,i])
        axs[i].set_ylabel(label)
        axs[i].grid()
    
    fig.tight_layout()
    plt.show()

