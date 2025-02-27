import torch

from torch import nn
from lietorch import SO3

class QuadrotorLoss(nn.Module):
    """
    Loss function for quadrotor states. The loss is a weighted sum of the following components:
    - position loss: MSE loss on position
    - velocity loss: MSE loss on velocity
    - angle loss: MSE loss on the angle between the rotation matrices
    - omega loss: MSE loss on the angular velocity
    """
    def __init__(self, reduce='mean', position_weight=1.0, velocity_weight=1.0, angle_weight=1000.0, omega_weight=1000.0):
        super(QuadrotorLoss, self).__init__()
        self._reduce = reduce
        self._position_weight = position_weight
        self._velocity_weight = velocity_weight
        self._angle_weight = angle_weight
        self._omega_weight = omega_weight

    def forward(self, input, target):
        """
        Compute the loss between the input and target states
        
        Parameters:
        ------------
        input: torch.Tensor
            Predicted states with shape (T, 13) or (B, T, 13)
        target: torch.Tensor
            Target states with shape (T, 13) or (B, T, 13)
        """
        position_loss = self._position_weight * nn.functional.mse_loss(input[...,0:3], target[...,0:3], reduction=self._reduce)
        velocity_loss = self._velocity_weight * nn.functional.mse_loss(input[...,3:6], target[...,3:6], reduction=self._reduce)
        # transform quaternions to rotation matrices and compute error in SO(3)
        R_input = SO3.InitFromVec(input[...,6:10].flatten(0,1))
        R_target = SO3.InitFromVec(target[...,6:10].flatten(0,1))
        dR = R_input.inv() * R_target
        angle_errors = dR.log().norm(dim=-1, keepdim=True)
        # Old code
        # R_input = roma.unitquat_to_rotmat(input[...,6:10])
        # R_target = roma.unitquat_to_rotmat(target[...,6:10])
        # angle_errors = 0.5 * vee_so3(R_target.transpose(-2,-1) @ R_input - R_input.transpose(-2,-1) @ R_target)
        if self._reduce == 'mean':
            angle_loss = torch.mean(angle_errors)
        else:
            angle_loss = angle_errors
        angle_loss = self._angle_weight * angle_loss
        omega_loss = self._omega_weight * nn.functional.mse_loss(input[...,10:13], target[...,10:13], reduction=self._reduce)
        
        return position_loss, velocity_loss, angle_loss, omega_loss
