from controller import ControllerLeeKhaled, state_t, vec3_s, attitude_t, setpoint_t, quaternion_t
import numpy as np
import rowan as rn

def skew(w):
    w = w.reshape(3,1)
    w1 = w[0,0]
    w2 = w[1,0]
    w3 = w[2,0]
    return np.array([[0, -w3, w2],[w3, 0, -w1],[-w2, w1, 0]]).reshape((3,3))


def computeDesiredRot(Fd, yaw):
    Rd = np.eye(3)
    normFd = np.linalg.norm(Fd)
    if normFd > 0:
        zdes = (Fd/normFd).reshape(3,)
    else:
      zdes = np.array([0,0,1])  
    xcdes = np.array([np.cos(yaw), np.sin(yaw), 0])
    normZX = np.linalg.norm(skew(zdes) @ xcdes)
    if normZX > 0:
        ydes = ((np.cross(zdes.reshape(3,), xcdes))/(normZX))
    else:
        ydes = np.array([0,1,0])
    xdes = np.cross(ydes.reshape(3,), zdes.reshape(3,))
    Rd[:,0] = xdes.reshape(3,)
    Rd[:,1] = ydes.reshape(3,)
    Rd[:,2] = zdes.reshape(3,)
    return Rd

def flatten(w_tilde):
    w1 = w_tilde[2,1]
    w2 = w_tilde[0,2]
    w3 = w_tilde[1,0]
    return np.array([w1,w2,w3])

def computeWd(m, R, T, desjerk):
    xb = R[:,0]
    yb = R[:,1]
    zb = R[:,2]
    if T == 0:
        hw = np.zeros(3,)
    else:
        hw = m/T * (desjerk - np.dot(zb, desjerk)*zb)
    p  = -np.dot(hw, yb)
    q  = np.dot(hw, xb)
    r  = 0
    return np.array([p,q,r])

def computeWddot(m, R, curr_w, T, Td, Td_dot, dessnap):
    xb = R[:,0]
    yb = R[:,1]
    zb = R[:,2]
    curr_w = curr_w.reshape(3,)
    if T == 0: 
        ha = np.zeros(3,)
    else:
        ha = (m/T)*dessnap - np.dot((Td_dot/T), zb) - (2/T)*np.cross(curr_w, np.dot(Td, zb)) \
            - np.cross(np.cross(curr_w, curr_w), zb) 
    return np.array([-np.dot(ha, yb), np.dot(ha, xb), 0])

def thrustCtrl(m, R ,refAcc, kpep, kvev):
    FdI = refAcc - kpep - kvev
    return (m * FdI.T @ R @ np.array([[0],[0],[1]]))[0,0], FdI

def torqueCtrl(I, Rt, curr_w_, krer, kwew, Rd, des_w): #, des_wd):
    curr_w = curr_w_.reshape((3,1))
    return ( -krer  - kwew + (np.cross(curr_w_, (I @ curr_w_))).reshape(3,1) \
        - I @ (skew(curr_w) @ Rt @ Rd @ des_w)).reshape(3,) #- Rt @ Rd @ des_wd) ).reshape(3,)

def controllerLee(uavModel, setpoint, state):
    
    kp      = uavModel.controller['kp']
    kv      = uavModel.controller['kd']
    kw      = uavModel.controller['kw']
    kr      = uavModel.controller['kr']
    
    currPos = np.array([state.position.x, state.position.y, state.position.z]).reshape((3,1))
    currVl  = np.array([state.velocity.x, state.velocity.y, state.velocity.z]).reshape((3,1))
    desPos = np.array([setpoint.position.x, setpoint.position.y, setpoint.position.z]).reshape((3,1))
    desVl  = np.array([setpoint.velocity.x, setpoint.velocity.y, setpoint.velocity.z]).reshape((3,1))    
    desAcc = np.array([setpoint.acceleration.x, setpoint.acceleration.y, setpoint.acceleration.z]).reshape((3,1))
    desjerk = np.array([setpoint.jerk.x, setpoint.jerk.y, setpoint.jerk.z]).reshape((3,))
    # dessnap = np.array([setpoint.snap.x, setpoint.snap.y, setpoint.snap.z]).reshape((3,))
    ep = (currPos - desPos)
    ev = (currVl  - desVl)
    m  = uavModel.m
    I  = uavModel.I

    gravComp = np.array([0,0,9.81]).reshape((3,1))
    R = rn.to_matrix(state.attitudeQuaternion)
    Rt = np.transpose(R)
    thrustSI, FdI = thrustCtrl(m, R, desAcc+gravComp, kp*ep, kv*ev)

    Rd  = computeDesiredRot(FdI,0)
    Rtd = np.transpose(Rd)
    
    er       = 0.5 * flatten((Rtd @ R - Rt @ Rd)).reshape((3,1)) 
    curr_w = np.array([state.attitudeRate.roll, state.attitudeRate.pitch, state.attitudeRate.yaw]).reshape((3,1))
    curr_w_  = curr_w.reshape(3,) # reshape of omega for cross products

    # zb      = Rd[:,2]
    T       = thrustSI#m * np.dot(FdI.reshape(3,), zb)
    # Td      = m * np.dot(desjerk, zb)
    des_w  = (computeWd(m, Rd, T, desjerk)).reshape((3,1))
    # des_w_ = des_w.reshape(3,)
    # Td_dot  = np.dot(zb, m * dessnap) - np.dot(zb, np.cross(np.cross(des_w_, des_w_), np.dot(T, zb)))
    # des_wd  = (computeWddot(m, R, des_w, T, Td, Td_dot, dessnap)).reshape(3,1)
    ew  = (curr_w - Rt @ Rd @ des_w).reshape((3,1))

    torque = torqueCtrl(I, Rt, curr_w_, kr*er, kw*ew, Rd, des_w) #, des_wd)
    
    return thrustSI, torque, des_w.reshape(3,) # , des_wd.reshape(3,)

class SimpleUAV:
    def __init__(self, m, I, kp=1., kd=1., kw=1., kr=1.):
        self.m = m
        self.I = I
        self.controller = {'kp':kp, 'kd':kd, 'kw':kw, 'kr': kr}

if __name__=="__main__":
    uavModel = SimpleUAV(m=0.034, I=np.diag([16.571710, 16.655602, 29.261652])*1e-6)
    controller = ControllerLeeKhaled(uavModel=uavModel)

    current_position = vec3_s(1.,1.,1.)
    current_velocity = vec3_s(0.,0.,0.)
    current_attitudeQuaternion = [1.,0.,0.,0.]
    current_attitudeRate = attitude_t(0.,0.,0.)
    current_state = state_t(position=current_position, velocity=current_velocity, attitudeQuaternion=current_attitudeQuaternion, attitudeRate=current_attitudeRate)
    desired_position = vec3_s(1.,1.,4.)
    desired_velocity = vec3_s(0.,0.,4.)
    desired_attitude = attitude_t(2.,0.,5.)
    # desired_attitudeRate = attitude_t(0.,0.,1.)
    desired_acceleration = vec3_s(0.,2.,4.)
    desired_jerk = vec3_s(0.,3.,4.)
    desired_setpoint = setpoint_t(
        position=desired_position, velocity=desired_velocity,
        attitude=desired_attitude, acceleration=desired_acceleration,
        jerk=desired_jerk
    )

    thrustSI, torque, des_w = controller.forward(current_state=current_state, desired_state=desired_setpoint)
    print("thrust: ", thrustSI, "\ntorque: ", torque, "\ndesired attitude rate: ", des_w)
    
    thrustSI, torque, des_w = controllerLee(uavModel=uavModel, state=current_state, setpoint=desired_setpoint)
    print("thrust: ", thrustSI, "\ntorque: ", torque, "\ndesired attitude rate: ", des_w)
