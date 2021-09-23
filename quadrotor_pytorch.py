import torch

# Quaternion routines adapted from rowan to use autograd
def qmultiply(q1, q2):
	return torch.cat((
		torch.tensor([q1[0] * q2[0] - torch.sum(q1[1:4]*q2[1:4])]),
		q1[0] * q2[1:4] + q2[0] * q1[1:4] + torch.cross(q1[1:4], q2[1:4])))

def qconjugate(q):
	return torch.cat((q[0:1],-q[1:4]))

def qrotate(q, v):
	quat_v = torch.cat((torch.tensor([0]), v))
	return qmultiply(q, qmultiply(quat_v, qconjugate(q)))[1:]

def qexp_regular_norm(q):
	e = torch.exp(q[0])
	norm = torch.linalg.norm(q[1:4])
	result_v = e * q[1:4] / norm * torch.sin(norm)
	result_w = e * torch.cos(norm)
	return torch.cat((torch.tensor([result_w]), result_v))

def qexp_zero_norm(q):
	e = torch.exp(q[0])
	result_v = torch.zeros(3)
	result_w = e
	return torch.cat((torch.tensor([result_w]), result_v))


def qexp(q):
	if torch.allclose(q[1:4], torch.zeros_like(q[1:4])):
		return qexp_zero_norm(q)
	else:
		return qexp_regular_norm(q)
	# return torch.where(torch.allclose(q[1:4], torch.zeros_like(q[1:4])), qexp_zero_norm(q), qexp_regular_norm(q))

def qintegrate(q, v, dt):
	quat_v = torch.cat((torch.tensor([0]), v*dt/2))
	return qmultiply(qexp(quat_v), q)

def qnormalize(q):
	return q / torch.linalg.norm(q)


class QuadrotorAutograd():

	def __init__(self):
		self.min_u = 0
		self.max_u = 12 / 1000 * 9.81

		self.min_x = torch.tensor(
					[-10, -10, -10, 					# Position
					  -3, -3, -3, 						# Velocity [m/s]
					  -1.001, -1.001, -1.001, -1.001,	# Quaternion
					  -35, -35, -35])	# angular velocity [rad/s]; CF sensor: +/- 2000 deg/s = +/- 35 rad/s
		self.max_x = -self.min_x

		# parameters (Crazyflie 2.0 quadrotor)
		self.mass = 0.034 # kg
		# self.J = np.array([
		# 	[16.56,0.83,0.71],
		# 	[0.83,16.66,1.8],
		# 	[0.72,1.8,29.26]
		# 	]) * 1e-6  # kg m^2
		self.J = torch.tensor([16.571710e-6, 16.655602e-6, 29.261652e-6])

		# Note: we assume here that our control is forces
		arm_length = 0.046 # m
		arm = 0.707106781 * arm_length
		t2t = 0.006 # thrust-to-torque ratio
		self.B0 = torch.tensor([
			[1, 1, 1, 1],
			[-arm, -arm, arm, arm],
			[-arm, arm, arm, -arm],
			[-t2t, t2t, -t2t, t2t]
			])
		self.g = 9.81 # not signed

		# if self.J.shape == (3,3):
		# 	self.inv_J = torch.linalg.pinv(self.J) # full matrix -> pseudo inverse
		# else:
		# 	self.inv_J = 1 / self.J # diagonal matrix -> division

		self.dt = 0.01


	def step(self, state, force):
		# compute next state
		q = state[6:10]
		omega = state[10:]

		eta = torch.mv(self.B0, force)
		# f_u = torch.tensor([0,0,eta[0]])
		f_u = torch.cat((torch.tensor([0,0]),eta[0:1]))
		# tau_u = torch.tensor([eta[1],eta[2],eta[3]])
		tau_u = eta[1:]

		# dynamics 
		# dot{p} = v 
		pos_next = state[0:3] + state[3:6] * self.dt
		# mv = mg + R f_u 
		vel_next = state[3:6] + (torch.tensor([0,0,-self.g]) + qrotate(q,f_u) / self.mass) * self.dt

		# dot{R} = R S(w)
		# to integrate the dynamics, see
		# https://www.ashwinnarayan.com/post/how-to-integrate-quaternions/, and
		# https://arxiv.org/pdf/1604.08139.pdf
		q_next = qnormalize(qintegrate(q, omega, self.dt))

		# mJ = Jw x w + tau_u
		inv_J = 1 / self.J  # diagonal matrix -> division
		omega_next = state[10:] + (inv_J * (torch.cross(self.J * omega,omega) + tau_u)) * self.dt

		return torch.cat((pos_next, vel_next, q_next, omega_next))


if __name__ == '__main__':

	robot = QuadrotorAutograd()

	xbar = torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], requires_grad=True, dtype=torch.float64)
	ubar = torch.tensor([0, 0, 0, 0], requires_grad=True, dtype=torch.float64)

	print(torch.autograd.gradcheck(robot.step, (xbar, ubar)))

	print(torch.autograd.functional.jacobian(robot.step, (xbar, ubar)))

	import timeit

	print(timeit.timeit(
		"""A, B = torch.autograd.functional.jacobian(robot.step, (xbar, ubar))""",
		globals=globals(),
		number=100))