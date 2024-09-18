import numpy as np
import rowan
import roma
import torch
from torch import nn
from quadrotor_pytorch import QuadrotorAutograd

# http://people.csail.mit.edu/jstraub/download/straubTransformationCookbook.pdf
def vee(R):
	assert R.shape == (3,3)
	# print(R)
	# exit()
	# return np.array([-R[1,2], R[0,2], -R[0,1]])
	return np.array([R[2,1], R[0,2], R[1,0]])

def hat_so3(v):
	return torch.tensor([[0., -v[2], v[1]],
					     [v[2], 0., -v[0]],
						 [-v[1], v[0], 0.]])

def vee_so3(R):
	return 0.5 * torch.tensor([[R[2,1]-R[1,2]],
							   [R[0,2]-R[2,0]],
							   [R[1,0]-R[0,1]]])

class vec3_s:
    def __init__(self, x=None, y=None, z=None):
        self.x = x
        self.y = y
        self.z = z

class attitude_t:
    def __init__(self, roll=None, pitch=None, yaw=None):
        self.roll  = roll
        self.pitch = pitch
        self.yaw   = yaw

class quaternion_t:
    def __init__(self, q0=None, q1=None, q2=None, q3=None, x=None, y=None, z=None, w=None):
        self.q0 = q0
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.x  = x 
        self.y  = y 
        self.z  = z 
        self.w  = w 

class state_t:
	def __init__(self, position=vec3_s(), velocity=vec3_s(), attitude=attitude_t(), attitudeQuaternion=quaternion_t(), attitudeRate=attitude_t(), acc=vec3_s(), payload_pos=vec3_s(), payload_vel=vec3_s()):
		self.position = position
		self.velocity = velocity
		self.attitude = attitude
		self.attitudeQuaternion = attitudeQuaternion
		self.attitudeRate = attitudeRate
		self.acc = acc
		self.payload_pos = payload_pos
		self.payload_vel = payload_vel

class mode:
	def __init__(self):
		self.x     = None
		self.y     = None
		self.z     = None
		self.roll  = None
		self.pitch = None
		self.roll  = None
		self.yaw   = None
		self.quat  = None

class setpoint_t:
	def __init__(self, position=vec3_s(), velocity=vec3_s(),
			  attitude=attitude_t(), attitudeQuaternion=quaternion_t(),
			  attitudeRate=attitude_t(), acceleration=vec3_s(), jerk=vec3_s(),
			  snap=vec3_s(), mode=mode()):
		self.position = position
		self.attitude = attitude
		self.attitudeQuaternion = attitudeQuaternion
		self.velocity = velocity
		self.attitudeRate = attitudeRate
		self.acceleration = acceleration
		self.jerk = jerk
		self.snap = snap
		self.mode = mode

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
		"""
		Takes the current state and the desired state as inputs and returns the controls.
		f is the total thrust and M are the control moments	
		"""
		
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

class ControllerLeeKhaled(nn.Module):
	def __init__(self, uavModel, kp=1., kv=1., kw=1., kr=1.):
		"""
		Parameters:
		-----------
		uavModel:
			model of the UAV to be controlled
		kp, kv, kw, kr: float
			gains of the controller
		"""
		super().__init__()
		self.uavModel = uavModel # the model the controller controls
		self.m = torch.tensor(uavModel.m, dtype=torch.double)
		self.I = torch.tensor(uavModel.I, dtype=torch.double)
	
		# make the gains tunable parameters
		self.kp = nn.Parameter(torch.tensor(kp))
		self.kv = nn.Parameter(torch.tensor(kv))
		self.kw = nn.Parameter(torch.tensor(kw))
		self.kr = nn.Parameter(torch.tensor(kr))

		self.double()
	
	def updateUAVModel(self, uavModel):
		self.m = torch.tensor(uavModel.m, dtype=torch.double)
		self.I = torch.tensor(uavModel.I, dtype=torch.double)
	
	def thrustCtrl(self, R, refAcc, ep, ev):
		"""
		Computes the total thrust f.

		Parameters:
		-----------
			R: torch.Tensor
				rotational matrix describing the attitude of the quadrotor
			refAcc: torch.Tensor
				sum of desired acceleration and gravity
			ep: torch.Tensor
				tracking error in position
			ev: torch.Tensor
				tracking error in velocity	
		
		Returns:
		--------
			thrustSI: torch.Tensor
				total thrust f
			FdI: torch.Tensor
				I don't know
		"""
		e3 = torch.tensor([0,0,1], dtype=torch.double).reshape((3,1))
		kpep = self.kp * ep
		kvev = self.kv * ev
		FdI = refAcc - kpep - kvev
		return (self.m * FdI.T @ R @ e3), FdI
	
	@staticmethod
	def computeDesiredRot(Fd, yaw):
		"""
		Computes the desired attitude.

		Parameters:
		-----------
			Fd: torch.Tensor
				desired force in b_3 direction
			yaw: torch.Tensor
				desired yaw of the quadrotor. relevant for b_1 vector
		"""
		Rd = torch.eye(3, dtype=torch.double)
		normFd = torch.linalg.norm(Fd)
		if normFd > 0:
			zdes = (Fd / normFd)
		else:
			zdes = torch.tensor([[0],[0],[1]], dtype=torch.double)
		xcdes = torch.tensor([torch.cos(yaw).item(), torch.sin(yaw).item(), 0], dtype=torch.double).reshape((3,1)) # 1,0,0 for yaw=0
		normZX = torch.linalg.norm(hat_so3(zdes) @ xcdes)
		if normZX > 0:
			ydes = torch.cross(zdes, xcdes) / normZX
		else:
			ydes = torch.tensor([0, 1, 0], dtype=torch.double).reshape((3,1))
		xdes = torch.cross(ydes, zdes)
		Rd[:,:1] = xdes
		Rd[:,1:2] = ydes
		Rd[:,2:3] = zdes
		return Rd

	def computeWd(self, R, T, desJerk):
		"""
		Computes the desired angular velocity omega.

		See Mellinger ad Kumar, 2011, equation (7) and following.

		Parameters:
		-----------
			R: torch.Tensor
				attitude of the quadrotor
			T: torch.Tensor
				total thrust for the quadrotor
			desJerk: torch.Tensor
				desired jerk of the quadrotor
		"""
		xb = R[:,0:1]
		yb = R[:,1:2]
		zb = R[:,2:3]
		if T == 0:
			hw = torch.zeros((3,1), dtype=torch.double)
		else:
			hw = self.m / T * (desJerk - zb.T @ desJerk * zb)
		p = -hw.T @ yb
		q = hw.T @ xb
		r = 0
		return torch.tensor([p, q, r], dtype=torch.double).reshape((3,1))


	def torqueCtrl(self, R, curr_w, er, ew, Rd, des_w): # , des_wd):
		krer = self.kr * er
		kwew = self.kw * ew
		return (-krer - kwew + (torch.cross(curr_w, (self.I @ curr_w)))) \
			- self.I @ (hat_so3(curr_w) @ R.T @ Rd @ des_w) # - R.T @ Rd @ des_wd)

	# the forward pass of the controller should be the main function
	# it takes a state and a desired state as input and returns the controls i.e. thrust and moments
	# the 'free' parameters of the controller are the gains allowing for optimization during training
	def forward(self, current_state: torch.Tensor, setpoint: torch.Tensor):
		"""
		Computes the desired controls for a current state given some target state. The controller uses terms up to the jerk but no snap.

		Parameters:
		-----------
			current_state: torch.Tensor
				Current state of the UAV. Contains position, attitude (unit quaternion), velocity and rotational velocity.
				[x, y, z, v_x, v_y, v_z, q_w, q_x, q_y, q_z, omega_roll, omega_pitch, omega_yaw]
			setpoint: torch.Tensor
				Setpoint ('desired state') of the trajectory. Contains position, velocity, acceleration, jerk and yaw
				[x, y, z, v_x, v_y, v_z, a_x, a_y, a_z, j_x, j_y, j_z, yaw]
		"""
		# current state of the quadrotor
		# currPos = torch.tensor([current_state.position.x, current_state.position.y, current_state.position.z]).reshape((3,1))
		currPos = current_state[:3].reshape((3,1))
		# currVel = torch.tensor([current_state.velocity.x, current_state.velocity.y, current_state.velocity.z]).reshape((3,1))
		currVel = current_state[3:6].reshape((3,1))
		R = roma.unitquat_to_rotmat(torch.tensor([current_state[7], current_state[8], current_state[9], current_state[6]]))
		# R = torch.tensor(rowan.to_matrix(current_state.attitudeQuaternion), dtype=torch.float)
		# currW = torch.tensor([current_state.attitudeRate.roll, current_state.attitudeRate.pitch, current_state.attitudeRate.yaw]).reshape((3,1))
		currW = current_state[10:13].reshape((3,1))

		# desired state of the quadrotor
		# desPos = torch.tensor([desired_state.position.x, desired_state.position.y, desired_state.position.z]).reshape((3,1))
		desPos = setpoint[:3].reshape((3,1))
		# desVel = torch.tensor([desired_state.velocity.x, desired_state.velocity.y, desired_state.velocity.z]).reshape((3,1))    
		desVel = setpoint[3:6].reshape((3,1))
		# desAcc = torch.tensor([desired_state.acceleration.x, desired_state.acceleration.y, desired_state.acceleration.z]).reshape((3,1))
		desAcc = setpoint[6:9].reshape((3,1))
		# desJerk = torch.tensor([desired_state.jerk.x, desired_state.jerk.y, desired_state.jerk.z]).reshape((3,1))
		desJerk = setpoint[9:12].reshape((3,1))
		desYaw = setpoint[12]
		# desSnap = torch.tensor([desired_state.snap.x, desired_state.snap.y, desired_state.snap.z]).reshape((3,1))
		ep = (currPos - desPos) # tracking error in position
		ev = (currVel  - desVel) # tracking error in velocity

		gravComp = torch.tensor([0.,0.,9.81], dtype=torch.double).reshape((3,1))
		thrustSI, FdI = self.thrustCtrl(R, desAcc+gravComp, ep, ev)

		Rd = self.computeDesiredRot(FdI, desYaw)

		er = 0.5 * vee_so3(Rd.T @ R - R.T @ Rd)

		# zb = Rd[:,2]
		T = thrustSI
		# Td = self.m * desJerk.T @ zb
		desW = self.computeWd(Rd, T, desJerk)
		# Td_dot = m * desSnap.T @ zb - # this line contains error in the original code
		# des_wd = self.computeWddot(R, des_w, T, Td, Td_dow, dessnap)
		ew = currW - R.T @ Rd @ desW
		torque = self.torqueCtrl(R, currW, er, ew, Rd, desW) #, des_wd)

		return thrustSI, torque, Rd, desW #, desWd # with the controls (thrustSI, torque), the current state and a proper model of the drone a next state can be computed
	
	def computeControl(self):
		return self.forward(self.uavModel.current_state, self.uavModel.desired_state)









if __name__ == '__main__':

	robot = QuadrotorAutograd()
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

