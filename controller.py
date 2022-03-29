import numpy as np
import rowan
from quadrotor_np import Quadrotor

# http://people.csail.mit.edu/jstraub/download/straubTransformationCookbook.pdf
def vee(R):
	assert R.shape == (3,3)
	# print(R)
	# exit()
	# return np.array([-R[1,2], R[0,2], -R[0,1]])
	return np.array([R[2,1], R[0,2], R[1,0]])

# Minimum Snap Trajectory Generation and Control for Quadrotors
# Daniel Mellinger and Vijay Kumar
# ICRA 2011
class ControllerMellinger:
	def __init__(self):
		# self.K_p = 6.0
		# self.K_v = 4.0
		# self.K_R = 10
		# self.K_omega = 0.0005
		self.K_p = 0.6
		self.K_v = 0.4
		self.K_R = -0.0001
		self.K_omega = 0.0001
		self.mass = 0

	def update(self, pos, vel, quat, omega, pos_des, vel_des, acc_des, omega_des, yaw_des):
		# print(quat)
		e_p = pos - pos_des
		e_v = vel - vel_des
		z_w = np.array([0,0,1])
		F_des = -self.K_p * e_p -self.K_v*e_v + self.mass * 9.81 * z_w + self.mass * acc_des

		z_B = rowan.rotate(quat, z_w)
		u1 = np.dot(F_des, z_B)

		z_Bdes = F_des / np.linalg.norm(F_des)
		x_Cdes = np.array([np.cos(yaw_des), np.sin(yaw_des), 0])
		y_Bdes = np.cross(z_Bdes, x_Cdes)
		y_Bdes = y_Bdes / np.linalg.norm(y_Bdes)
		x_Bdes = np.cross(y_Bdes, z_Bdes)
		R_des = np.vstack([x_Bdes, y_Bdes, z_Bdes])
		q_des = rowan.from_matrix(R_des)
		R_B = rowan.to_matrix(quat)

		# print(z_B, R_B[:,2])

		e_R = 0.5 * vee(R_des.T @ R_B - R_B.T @ R_des)
		# print(e_R)

		# #
		# e_R[1] = -e_R[1]

		# TODO: omega_des needs to be in body frame coordinates!
		e_omega = omega - omega_des

		u2, u3, u4 = -self.K_R * e_R - self.K_omega * e_omega

		return np.array([u1, u2, u3, u4]), q_des


# Geometric Tracking Control of a Quadrotor UAV on SE(3)
# Taeyoung Lee, Melvin Leok, and N. Harris McClamroch
# CDC 2010
class ControllerLee:
	def __init__(self):
		# self.K_p = 6.0
		# self.K_v = 4.0
		# self.K_R = 10
		# self.K_omega = 0.0005
		self.k_x = 0.6
		self.k_v = 0.4
		self.k_R = 0.0001
		self.k_omega = 0.0001
		self.mass = 0

	def update(self, pos, vel, quat, omega, pos_des, vel_des, acc_des, omega_des, yaw_des):
		
		e_x = pos - pos_des # (6)
		e_v = vel - vel_des # (7)

		e3 = np.array([0, 0, 1])

		# (12)
		b_1d = np.array([np.cos(yaw_des), np.sin(yaw_des), 0])
		b_3d = -self.k_x*e_x - self.k_v*e_v - self.mass * 9.81 * e3 + self.mass * acc_des
		b_3d = -b_3d / np.linalg.norm(b_3d)
		b_2d = np.cross(b_3d, b_1d)
		R_d = np.vstack([np.cross(b_2d, b_3d), b_2d, b_3d])
		q_d = rowan.from_matrix(R_d)

		# (15)
		R = rowan.to_matrix(quat)
		f = np.dot(-(-self.k_x*e_x-self.k_v*e_v-self.mass*9.81*e3+self.mass*acc_des), R @ e3)

		# (10)
		e_R = 0.5 * vee(R_d.T @ R - R.T @ R_d)

		# (11)
		e_omega = omega - R.T @ R_d @ omega_des

		# (16)
		# TODO: add higher order terms!
		M = -self.k_R * e_R - self.k_omega * e_omega

		return np.array([f, M[0], M[1], M[2]]), q_d


if __name__ == '__main__':

	robot = Quadrotor()
	robot.dt = 0.001
	# controller = ControllerMellinger()
	controller = ControllerLee()
	controller.mass = robot.mass
	B0_inv = np.linalg.inv(robot.B0)

	q0 = rowan.from_euler(np.radians(0), np.radians(0), np.radians(0), 'xyz')

	x0 = np.array([0.00, 0.00, 0.0, 0, 0, 0, q0[0], q0[1], q0[2], q0[3], 0, 0, 0], dtype=np.float32)
	xf = np.array([0.00, 0.10, 0.0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
	# xf = np.array([0.00, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)

	states = [x0]
	actions = []
	states_desired = []

	for t in np.arange(0, 0.5, robot.dt):
		x = states[-1]
		eta, q_des = controller.update(x[0:3], x[3:6], x[6:10], x[10:13], xf[0:3], xf[3:6], np.zeros(3), np.zeros(3), 0)
		u = B0_inv @ eta
		# u = np.ones(4)
		u = np.clip(u, robot.min_u, robot.max_u)
		# print(u)
		x_next = robot.step(x, u)
		# print(x_next)
		states.append(x_next)
		actions.append(u)
		states_desired.append(np.concatenate((xf[0:3], xf[3:6], q_des, np.zeros(3))))

	# store the last result
	data = np.empty((len(states), 13+4+13), dtype=np.float32)
	data[:, 0:13] = states
	data[:-1, 13:17] = actions
	data[-1, 13:17] = np.nan
	data[:-1, 17:] = states_desired
	data[-1, 17:] = np.nan

	np.save("data.npy", data, allow_pickle=False, fix_imports=False)

