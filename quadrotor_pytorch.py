import torch

# Quaternion routines adapted from rowan to use autograd
def qmultiply(q1, q2):
	t1 = torch.mul(q1[...,:1], q2[...,:1]) - torch.sum(torch.mul(q1[...,1:4],q2[...,1:4]), dim=-1, keepdim=True)
	t2 = q1[...,:1] * q2[...,1:4] + q2[...,:1] * q1[...,1:4] + torch.cross(q1[...,1:4], q2[...,1:4], dim=-1)
	return torch.cat((t1, t2), dim=-1)

def qconjugate(q):
	return torch.cat((q[...,0:1],-q[...,1:4]), dim=-1)

def qrotate(q, v):
	quat_v = torch.cat((torch.zeros((v.shape[0],1)), v), dim=-1)
	return qmultiply(q, qmultiply(quat_v, qconjugate(q)))[...,1:]

def qexp_regular_norm(q):
	e = torch.exp(q[...,:1])
	norm = torch.linalg.norm(q[...,1:4], dim=-1, keepdim=True)
	result_v = e * q[...,1:4] / norm * torch.sin(norm)
	result_w = e * torch.cos(norm)
	return torch.cat((result_w, result_v), dim=-1)

def qexp_zero_norm(q):
	e = torch.exp(q[...,:1])
	result_v = torch.zeros((q.shape[0], 3))
	result_w = e
	return torch.cat((result_w, result_v), dim=-1)

def qexp(q):
	"""
	Adopted from rowan
	"""
	expo = torch.empty_like(q)
	norms = torch.linalg.norm(q[..., 1:], dim=-1)
	e = torch.exp(q[..., 0])
	expo[...,0] = e * torch.cos(norms)
	norm_zero = torch.isclose(norms, torch.zeros_like(norms))
	not_zero = torch.logical_not(norm_zero)

	if torch.any(not_zero):
		expo[not_zero,1:] = (
			e[not_zero, torch.newaxis]
			* (q[not_zero, 1:] / norms[not_zero, torch.newaxis])
			* torch.sin(norms)[not_zero, torch.newaxis]
		)
		if torch.any(norm_zero):
			expo[norm_zero, 1:] = 0
	else:
		expo[..., 1:] = 0
	
	return expo

def qintegrate(q, v, dt):
	quat_v = torch.cat((torch.zeros((v.shape[0],1)), v*dt/2),dim=-1)
	return qmultiply(qexp(quat_v), q)

def qnormalize(q):
	return q / torch.linalg.norm(q, dim=-1, keepdim=True)


class QuadrotorAutograd():

	def __init__(self, noise_on=False, mass=0.034, inertia=[16.571710e-6, 16.655602e-6, 29.261652e-6]):
		# control limits in Newtons (same for each rotor)
		self.min_u = 0
		self.max_u = 12 / 1000 * 9.81

		# feasible state space
		self.min_x = torch.tensor(
					[-10, -10, -10, 					# Position
					  -3, -3, -3, 						# Velocity [m/s]
					  -1.001, -1.001, -1.001, -1.001,	# Quaternion
					  -35, -35, -35])	# angular velocity [rad/s]; CF sensor: +/- 2000 deg/s = +/- 35 rad/s
		self.max_x = -self.min_x

		# parameters (Crazyflie 2.0 quadrotor)
		self.m = torch.tensor(mass, dtype=torch.double) # true mass in kg
		# self.I = np.array([
		# 	[16.56,0.83,0.71],
		# 	[0.83,16.66,1.8],
		# 	[0.72,1.8,29.26]
		# 	]) * 1e-6  # kg m^2
		self.I = torch.tensor(inertia, dtype=torch.double)
		# self.I = torch.tensor([1.05, 1.0, .95], dtype=torch.float64)

		# Note: we assume here that our control is forces
		arm_length = 0.046 # m
		arm = 0.707106781 * arm_length
		t2t = 0.006 # thrust-to-torque ratio

		# control forces per rotor to thrust+force vector 
		self.B0 = torch.tensor([
			[1, 1, 1, 1],
			[-arm, -arm, arm, arm],
			[-arm, arm, arm, -arm],
			[-t2t, t2t, -t2t, t2t]
			], dtype=torch.float64)
		self.g = 9.81 # not signed

		self.kf = 2.1

		# if self.I.shape == (3,3):
		# 	self.inv_I = torch.linalg.pinv(self.I) # full matrix -> pseudo inverse
		# else:
		# 	self.inv_I = 1 / self.I # diagonal matrix -> division

		self.dt = 0.01
		self.noise_on = noise_on

	def _rpm_to_force(self, controls):
		return self.kf * 1e-10 * torch.pow(controls,2)  # motor controls to force

	def step(self, state, force, dt):
		# compute next state
		q = state[...,6:10]
		omega = state[...,10:]

		# clip for to control limits
		force = torch.clamp(force, self.min_u, self.max_u)

		eta = force @ self.B0.T
		batch_size = state.shape[0]
		f_u = torch.cat((torch.zeros((batch_size, 2)),eta[...,:1]),dim=-1)
		tau_u = eta[...,1:]

		# dynamics 
		# dot{p} = v 
		pos_next = state[...,:3] + state[...,3:6] * dt
		if self.noise_on:
			with torch.no_grad():
				pos_next += state[...,3:6] * torch.randn_like(state[..., 3:6]) * dt
		# mv = mg + R f_u 
		vel_next = state[...,3:6] + (torch.tensor([0,0,-self.g]) + qrotate(q,f_u) / self.m) * dt

		# dot{R} = R S(w)
		# to integrate the dynamics, see
		# https://www.ashwinnarayan.com/post/how-to-integrate-quaternions/, and
		# https://arxiv.org/pdf/1604.08139.pdf
		omega_global = qrotate(q, omega)
		q_next = qnormalize(qintegrate(q, omega_global, dt))

		# mI = Iw x w + tau_u
		inv_I = 1 / self.I  # diagonal matrix -> division
		omega_next = state[...,10:] + (inv_I * (torch.cross(self.I * omega,omega, dim=-1) + tau_u)) * dt

		return torch.cat((pos_next, vel_next, q_next, omega_next), dim=-1)


if __name__ == '__main__':

	robot = QuadrotorAutograd()

	# state: x = [x,y,z, v_x, v_y, v_x, q_w, q_x, q_y, q_z, omega_roll, omega_pitch, omega_yaw]
	xbar = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], requires_grad=True, dtype=torch.float64)
	ubar = torch.tensor([[0, 0, 0, 0]], requires_grad=True, dtype=torch.float64)

	print(torch.autograd.gradcheck(robot.step, (xbar, ubar)))

	# print(torch.autograd.functional.jacobian(robot.step, (xbar, ubar)))

	import timeit

	print(timeit.timeit(
		"""A, B = torch.autograd.functional.jacobian(robot.step, (xbar, ubar))""",
		globals=globals(),
		number=100))
