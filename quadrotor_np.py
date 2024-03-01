import numpy as np
import rowan

class Quadrotor():

	def __init__(self):
		self.min_u = 0
		self.max_u = 12 / 1000 * 9.81

		self.min_x = np.array(
					[-10, -10, -10, 					# Position
					  -3, -3, -3, 						# Velocity [m/s]
					  -1.001, -1.001, -1.001, -1.001,	# Quaternion
					  -35, -35, -35], dtype=np.float32)	# angular velocity [rad/s]; CF sensor: +/- 2000 deg/s = +/- 35 rad/s
		self.max_x = -self.min_x

		# parameters (Crazyflie 2.0 quadrotor)
		self.mass = 0.034 # kg
		# self.J = np.array([
		# 	[16.56,0.83,0.71],
		# 	[0.83,16.66,1.8],
		# 	[0.72,1.8,29.26]
		# 	]) * 1e-6  # kg m^2
		self.J = np.array([16.571710e-6, 16.655602e-6, 29.261652e-6])

		# Note: we assume here that our control is forces
		arm_length = 0.046 # m
		arm = 0.707106781 * arm_length
		t2t = 0.006 # thrust-to-torque ratio
		self.B0 = np.array([
			[1, 1, 1, 1],
			[-arm, -arm, arm, arm],
			[-arm, arm, arm, -arm],
			[-t2t, t2t, -t2t, t2t]
			])
		self.g = 9.81 # not signed

		if self.J.shape == (3,3):
			self.inv_J = np.linalg.pinv(self.J) # full matrix -> pseudo inverse
		else:
			self.inv_J = 1 / self.J # diagonal matrix -> division

		self.dt = 0.01


	def step(self, state, force):
		# compute next state
		pos = state[0:3]
		vel = state[3:6]
		q = state[6:10]
		omega = state[10:]

		eta = self.B0 @ force
		f_u = np.array([0,0,eta[0]])
		tau_u = np.array([eta[1],eta[2],eta[3]])

		# dynamics 
		# dot{p} = v 
		pos_next = pos + vel * self.dt
		# mv = mg + R f_u 
		vel_next = vel + (np.array([0,0,-self.g]) + rowan.rotate(q,f_u) / self.mass) * self.dt

		# dot{R} = R S(w)
		# to integrate the dynamics, see
		# https://www.ashwinnarayan.com/post/how-to-integrate-quaternions/, and
		# https://arxiv.org/pdf/1604.08139.pdf
		omega_global = rowan.rotate(q, omega)
		q_next = rowan.normalize(rowan.calculus.integrate(q, omega_global, self.dt))

		# mJ = Jw x w + tau_u 
		omega_next = omega + (self.inv_J * (np.cross(self.J * omega,omega) + tau_u)) * self.dt

		return np.concatenate((pos_next, vel_next, q_next, omega_next))


if __name__ == '__main__':

	pass