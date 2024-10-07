import roma
import torch

def hat_so3(v: torch.Tensor):
    Rs = torch.zeros((*v.shape[:-2],3,3), dtype=v.dtype)
    Rs[...,0,1] = -v[...,2,0]
    Rs[...,0,2] = v[...,1,0]
    Rs[...,1,0] = v[...,2,0]
    Rs[...,1,2] = -v[...,0,0]
    Rs[...,2,0] = -v[...,1,0]
    Rs[...,2,1] = v[...,0,0]
    return Rs

def vee_so3(R:torch.Tensor):
    return 0.5 * torch.stack([R[...,2,1]-R[...,1,2],
                              R[...,0,2]-R[...,2,0],
                              R[...,1,0]-R[...,0,1]], axis=-1)

class ControllerLee():
    def __init__(self, kp=1., kv=1., kw=1., kr=1., mass=1., inertia=[1.,1.,1.]):
        """
        Parameters:
        -----------
        kp, kv, kw, kr: float
            gains of the controller
        mass: float
            mass of the UAV to be controlled
        inertia: array
            inertia of the UAV to be controlled
        """
        super().__init__()
        if isinstance(mass, torch.Tensor):
            self.m = mass.clone().detach()
        else:
            self.m = torch.tensor(mass)
        
        if isinstance(inertia, torch.Tensor):
            self.I = inertia.clone().detach()
        else:
            self.I = torch.tensor(inertia)

        if self.I.dim() == 1:
            self.I = torch.diag(self.I)
    
        # make the gains tunable parameters
        self.kp = torch.tensor(kp)
        self.kv = torch.tensor(kv)
        self.kw = torch.tensor(kw)
        self.kr = torch.tensor(kr)

    def thrustCtrl(self, R:torch.Tensor, refAcc:torch.Tensor, ep:torch.Tensor, ev:torch.Tensor):
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
        e3 = torch.tensor([[0.],[0.],[1.]], dtype=R.dtype)
        kpep = self.kp * ep
        kvev = self.kv * ev
        FdI = refAcc - kpep - kvev
        return (self.m * FdI.transpose(1,2) @ R @ e3), FdI
    
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
        assert Fd.dim() == 3, "Missing batch dimension for Fd"
        batch_size = Fd.shape[0]
        Rd = torch.empty((batch_size,3,3), dtype=Fd.dtype)
        normFd = torch.linalg.norm(Fd, dim=(-2,-1),keepdim=True)
        zdes = torch.zeros_like(Fd)
        zdes_mask = normFd.squeeze((-2,-1)) > 0
        zdes[zdes_mask] = (Fd / normFd)[zdes_mask]
        zdes[~zdes_mask] = torch.tensor([[0],[0],[1]], dtype=Fd.dtype)
        
        xcdes = torch.zeros((batch_size,3,1), dtype=Fd.dtype)
        xcdes[:,0,0] = torch.cos(yaw)
        xcdes[:,1,0] = torch.sin(yaw)
        normZX = torch.linalg.norm(hat_so3(zdes) @ xcdes, dim=(-2,-1))

        ydes = torch.zeros_like(Fd)
        ydes_mask = normZX > 0
        ydes[ydes_mask] = torch.cross(zdes, xcdes, dim=-2)[ydes_mask]
        ydes[~ydes_mask] = torch.tensor([[0],[1],[0]], dtype=Fd.dtype)
        
        xdes = torch.cross(ydes, zdes, dim=-2)
        Rd[:,:,:1] = xdes
        Rd[:,:,1:2] = ydes
        Rd[:,:,2:3] = zdes
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
        batch_size = R.shape[0]
        xb = R[:,:,0:1]
        yb = R[:,:,1:2]
        zb = R[:,:,2:3]
        hw = torch.zeros_like(desJerk)
        # TODO: maybe this should test for a range close to zeros because of numerical issues
        hw_mask = (T==0).squeeze((-2,-1))
        hw[~hw_mask] = self.m / T[~hw_mask] * (desJerk[~hw_mask] - zb[~hw_mask].mT @ desJerk[~hw_mask] * zb[~hw_mask])
        p = -hw.mT @ yb
        q = hw.mT @ xb
        r = torch.zeros((batch_size,1,1), dtype=desJerk.dtype)
        return torch.concat([p,q,r], dim=1)


    def torqueCtrl(self, R, curr_w, er, ew, Rd, des_w): # , des_wd):
        krer = self.kr * er
        kwew = self.kw * ew
        return (-krer - kwew + (torch.cross(curr_w, (self.I @ curr_w), dim=-2))) \
            - self.I @ (hat_so3(curr_w) @ R.mT @ Rd @ des_w) # - R.T @ Rd @ des_wd)

    # the forward pass of the controller should be the main function
    # it takes a state and a desired state as input and returns the controls i.e. thrust and moments
    # the 'free' parameters of the controller are the gains allowing for optimization during training
    def compute_controls(self, current_state: torch.Tensor, setpoint: torch.Tensor):
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
        currPos = current_state[:,:3].reshape((-1,3,1))
        currVel = current_state[:,3:6].reshape((-1,3,1))
        R = roma.unitquat_to_rotmat(torch.stack([current_state[:,7], current_state[:,8], current_state[:,9], current_state[:,6]],axis=-1))
        currW = current_state[:,10:13].reshape((-1,3,1))

        # desired state of the quadrotor
        desPos = setpoint[:,:3].reshape((-1,3,1))
        desVel = setpoint[:,3:6].reshape((-1,3,1))
        desAcc = setpoint[:,6:9].reshape((-1,3,1))
        desJerk = setpoint[:,9:12].reshape((-1,3,1))
        desYaw = setpoint[:,12]
        # desSnap = torch.tensor([desired_state.snap.x, desired_state.snap.y, desired_state.snap.z]).reshape((3,1))
        ep = (currPos - desPos) # tracking error in position
        ev = (currVel  - desVel) # tracking error in velocity

        gravComp = torch.tensor([[0.],[0.],[9.81]], dtype=ep.dtype)
        thrustSI, FdI = self.thrustCtrl(R, desAcc+gravComp, ep, ev)

        Rd = self.computeDesiredRot(FdI, desYaw)

        er = 0.5 * vee_so3(Rd.transpose(1,2) @ R - R.transpose(1,2) @ Rd).unsqueeze(-1) # tracking error in rotation

        # zb = Rd[:,2]
        T = thrustSI
        # Td = self.m * desJerk.T @ zb
        desW = self.computeWd(Rd, T, desJerk)
        # Td_dot = m * desSnap.T @ zb - # this line contains error in the original code
        # des_wd = self.computeWddot(R, des_w, T, Td, Td_dow, dessnap)
        ew = currW - R.transpose(1,2) @ Rd @ desW
        torque = self.torqueCtrl(R, currW, er, ew, Rd, desW) #, des_wd)

        return thrustSI, torque, Rd, desW #, desWd # with the controls (thrustSI, torque), the current state and a proper model of the drone a next state can be computed