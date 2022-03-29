import numpy as np
import subprocess
import time
import rowan
import argparse

import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages

# visualization related
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

from quadrotor_jax import QuadrotorAutograd

def animate(data):
	vis = meshcat.Visualizer()
	vis.open()

	vis["/Cameras/default"].set_transform(
		tf.translation_matrix([0, 0, 0]).dot(
		tf.euler_matrix(0, np.radians(-30), -np.pi/2)))

	vis["/Cameras/default/rotated/<object>"].set_transform(
		tf.translation_matrix([1, 0, 0]))

	vis["Quadrotor"].set_object(
		g.StlMeshGeometry.from_file('../scp-baseline/crazyflie2.stl'))

	while True:
		for row in data:
			vis["Quadrotor"].set_transform(
				tf.translation_matrix([row[0], row[1], row[2]]).dot(
					tf.quaternion_matrix(row[6:10])))
			time.sleep(0.1)


def generatePDF(data, filename):
	pp = PdfPages(filename)

	fig, axs = plt.subplots(2, 3, sharex='all', sharey='row')
	for k, name in enumerate(['x', 'y', 'z']):
		axs[0, k].plot(data[:, k])
		axs[0, k].plot(data[:, 17+k], '--')
		axs[0, k].set_ylabel(name + " [m]")

	for k, name in enumerate(['vx', 'vy', 'vz']):
		axs[1, k].plot(data[:, 3+k])
		axs[1, k].plot(data[:, 17+3+k], '--')
		axs[1, k].set_ylabel(name + " [m/s]")
		axs[1, k].set_xlabel("timestep")

	pp.savefig(fig)
	plt.close(fig)

	fig, axs = plt.subplots(2, 3, sharex='all', sharey='row')
	rpy = np.degrees(rowan.to_euler(data[:, 6:10], 'xyz'))
	rpy_des = np.degrees(rowan.to_euler(data[:-1, 17+6:17+10], 'xyz'))
	for k, name in enumerate(['roll', 'pitch', 'yaw']):
		axs[0, k].plot(rpy[:, k])
		axs[0, k].plot(rpy_des[:, k], '--')
		axs[0, k].set_ylabel(name + " [deg]")

	# try to numerically estimate omega_des
	# see https://math.stackexchange.com/questions/2282938/converting-from-quaternion-to-angular-velocity-then-back-to-quaternion
	est = []
	dt = 0.001
	for t in range(0, data.shape[0]-2):
		omega_est = 2 * rowan.multiply(rowan.conjugate(data[t, 17+6:17+10]), data[t+1, 17+6:17+10])[1:4] / dt
		est.append(omega_est)
	est = np.array(est)
	print(est)

	for k, name in enumerate(['wx', 'wy', 'wz']):
		axs[1, k].plot(np.degrees(data[:, 10+k]))
		axs[1, k].plot(np.degrees(est[:,k]), '--')
		axs[1, k].set_ylabel(name + " [deg/s]")
		axs[1, k].set_xlabel("timestep")

	pp.savefig(fig)
	plt.close(fig)

	fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')

	for k, name in enumerate(['m1', 'm2', 'm3', 'm4']):
		axs[k//2,k%2].plot(data[:-1, 13 + k])
		axs[k//2, k % 2].set_ylabel(name + " [N]")
		axs[1, k % 2].set_xlabel("timestep")
		axs[k//2, k % 2].axhline(12./1000.*9.81)
	
	pp.savefig(fig)
	plt.close(fig)

	fig, ax = plt.subplots()
	ax.plot(np.sum(data[:-1, 13:17], axis=1))
	ax.set_title('actions sum')
	pp.savefig(fig)
	plt.close(fig)

	pp.close()
	subprocess.call(["xdg-open", filename])


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("file")
	args = parser.parse_args()

	dt = 0.01

	data = np.load(args.file)
	data[:, 6:10] = rowan.normalize(data[:, 6:10])
	# data[:, 17+6:17+10] = rowan.normalize(data[:, 17+6:17+10])

	generatePDF(data, 'output.pdf')
	# animate(data_propagated)
