import argparse

from torch.utils.data.dataset import TensorDataset
from data import cfusdlog
from quadrotor_pytorch import QuadrotorAutograd 

import torch
from torch import nn
from torch.utils.data import DataLoader


class QuadrotorModule(nn.Module):
    def __init__(self, dt):
        super(QuadrotorModule, self).__init__()
        self.quad = QuadrotorAutograd()
        self.quad.dt = dt

        # optimize mass
        self.mass = nn.Parameter(torch.tensor([self.quad.mass]))
        self.quad.mass = self.mass


        # self.J = nn.Parameter(self.quad.J)
        # self.quad.J = self.J

        # self.B0 = nn.Parameter(self.quad.B0)
        # self.quad.B0 = self.B0

        self.kf = 2.1
        self.double()
        # self.kf = nn.Parameter(torch.tensor([self.kf]))

    def forward(self, x):
        # print(x.shape)
        state = x[0,0:13]
        # kf = 1e-10
        # print(self.kf, x)
        force = self.kf * 1e-10 * torch.pow(x[0, 13:], 2)
        # force = self.kf * x[0, 13:]
        # print(force)
        # exit()
        # print(x[0,13:], force)
        # exit()
        # print(torch.sum(force))
        next_state = self.quad.step(state, force)

        # print(state)
        # print(force)
        # print(next_state)
        # exit()
        # print(next_state)
        return next_state.reshape((1,13))


# quaternion norm (adopted from rowan)
def qnorm(q):
    return torch.linalg.norm(q, axis=-1)

# quaternion sym distance (adopted from rowan)
def qsym_distance(p, q):
    return torch.minimum(qnorm(p - q), qnorm(p + q))

class QuadrotorLoss(nn.Module):
    def __init__(self):
        super(QuadrotorLoss, self).__init__()

    def forward(self, input, target):
        # print(input, target)
        position_loss = torch.nn.functional.mse_loss(input[:,0:3], target[:,0:3])
        velocity_loss = torch.nn.functional.mse_loss(input[:,3:6], target[:,3:6])
        angle_errors = qsym_distance(input[:, 6:10], target[:, 6:10])
        angle_loss = torch.mean(angle_errors)
        omega_loss = torch.nn.functional.mse_loss(input[:,10:13], target[:,10:13])
        # return position_loss + velocity_loss + angle_loss + omega_loss
        return velocity_loss

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    training_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    training_loss /= size
    print(f"Training Error: \n Avg loss: {training_loss:>8f} \n")


def test_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


# pwm normalized [0-1]; vbat normalized [0-1]
# output in N
def pwm2force(pwm, vbat):
    C_00 = 11.093358483549203
    C_10 = -39.08104165843915
    C_01 = -9.525647087583181
    C_20 = 20.573302305476638
    C_11 = 38.42885066644033
    return (C_00 + C_10*pwm + C_01*vbat + C_20*pwm**2 + C_11*vbat*pwm) / 1000 * 9.81


def load(filename):
    # decode binary log data
    data_usd = cfusdlog.decode(filename)

    T = len(data_usd['fixedFrequency']['timestamp'])

    dts = torch.diff(torch.from_numpy(data_usd['fixedFrequency']['timestamp']))
    dt = torch.mean(dts).item() / 1000

    data_torch = torch.empty((T, 13+4), dtype=torch.float64)

    data_torch[:, 0] = torch.from_numpy(
        data_usd['fixedFrequency']['stateEstimateZ.x']) / 1000.0
    data_torch[:, 1] = torch.from_numpy(
        data_usd['fixedFrequency']['stateEstimateZ.y']) / 1000.0
    data_torch[:, 2] = torch.from_numpy(
        data_usd['fixedFrequency']['stateEstimateZ.z']) / 1000.0
    data_torch[:, 3] = torch.from_numpy(
        data_usd['fixedFrequency']['stateEstimateZ.vx']) / 1000.0
    data_torch[:, 4] = torch.from_numpy(
        data_usd['fixedFrequency']['stateEstimateZ.vy']) / 1000.0
    data_torch[:, 5] = torch.from_numpy(
        data_usd['fixedFrequency']['stateEstimateZ.vz']) / 1000.0
    for t in range(T):
        # q is in [x,y,z,w] format
        q = cfusdlog.quatdecompress(
            data_usd['fixedFrequency']['stateEstimateZ.quat'][t])
        # [w,x,y,z] format
        data_torch[t, 6] = torch.from_numpy(q[3:])
        data_torch[t, 7:10] = torch.from_numpy(q[0:3])
    data_torch[:, 10] = torch.from_numpy(
        data_usd['fixedFrequency']['stateEstimateZ.rateRoll']) / 1000.0
    data_torch[:, 11] = torch.from_numpy(
        data_usd['fixedFrequency']['stateEstimateZ.ratePitch']) / 1000.0
    data_torch[:, 12] = torch.from_numpy(
        data_usd['fixedFrequency']['stateEstimateZ.rateYaw']) / 1000.0

    data_torch[:, 13] = torch.from_numpy(data_usd['fixedFrequency']['rpm.m1'])
    data_torch[:, 14] = torch.from_numpy(data_usd['fixedFrequency']['rpm.m2'])
    data_torch[:, 15] = torch.from_numpy(data_usd['fixedFrequency']['rpm.m3'])
    data_torch[:, 16] = torch.from_numpy(data_usd['fixedFrequency']['rpm.m4'])
    # vbat_norm = data_usd['fixedFrequency']['asc37800.v_mV'] / 1000 / 4.2

    # data_torch[:, 13] = torch.from_numpy(pwm2force(data_usd['fixedFrequency']['pwm.m1_pwm'] / 65536, vbat_norm))
    # data_torch[:, 14] = torch.from_numpy(pwm2force(data_usd['fixedFrequency']['pwm.m2_pwm'] / 65536, vbat_norm))
    # data_torch[:, 15] = torch.from_numpy(pwm2force(data_usd['fixedFrequency']['pwm.m3_pwm'] / 65536, vbat_norm))
    # data_torch[:, 16] = torch.from_numpy(pwm2force(data_usd['fixedFrequency']['pwm.m4_pwm'] / 65536, vbat_norm))


    # import matplotlib.pyplot as plt
    # import rowan
    # import numpy as np

    # fig, ax = plt.subplots(4,1)
    # ax[0].plot(np.diff(data_torch[:,12].numpy()))
    # # ax[0].plot(data_torch[:,13].numpy())
    # # ax[1].plot(data_torch[:,14].numpy())
    # # ax[2].plot(data_torch[:,15].numpy())
    # # ax[3].plot(data_torch[:,16].numpy())
    # # rpy = np.degrees(rowan.to_euler(data_torch[:, 6:10].numpy(), 'xyz'))
    # # ax[0].plot(rpy[:,2])
    # # ax[0].plot(data_usd['estPose']['timestamp'], data_usd['estPose']['locSrv.qx'])
    # # ax[0].plot(data_usd['fixedFrequency']['timestamp'], data_torch[:, 6].numpy())
    # # ax[1].plot(data_usd['estPose']['timestamp'], data_usd['estPose']['locSrv.qy'])
    # # ax[1].plot(data_usd['fixedFrequency']['timestamp'], data_torch[:, 7].numpy())
    # # ax[2].plot(data_usd['estPose']['timestamp'], data_usd['estPose']['locSrv.qz'])
    # # ax[2].plot(data_usd['fixedFrequency']['timestamp'], data_torch[:, 8].numpy())
    # # ax[3].plot(data_usd['estPose']['timestamp'], data_usd['estPose']['locSrv.qw'])
    # # ax[3].plot(data_usd['fixedFrequency']['timestamp'], data_torch[:, 9].numpy())
    # # ax.set_xlabel("position [m]")
    # # ax.set_ylabel("velocity [m/s]")

    # plt.show()

    # exit()

    # input is 13-dimensional state and action (motor rpm)
    x = data_torch[0:-1]
    # label is 13 dimensional "next" state after applying the action
    y = data_torch[1:, 0:13]

    training_data = TensorDataset(x, y)
    return dt, training_data

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("file_usd_train")
    parser.add_argument("file_usd_test")
    args = parser.parse_args()

    dt, training_data = load(args.file_usd_train)
    dt2, test_data = load(args.file_usd_test)

    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=1)


    model = QuadrotorModule(dt)

    # loss_fn = nn.MSELoss()
    loss_fn = QuadrotorLoss()

    learning_rate = 1e-3
    epochs = 10

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")
    print(model.state_dict())




