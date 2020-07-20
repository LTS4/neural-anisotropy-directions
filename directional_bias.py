import numpy as np

import torch
import torch.utils.data as data

from torch.utils.data import DataLoader

from utils import train

class DirectionalLinearDataset(data.Dataset):

    def __init__(self,
                 v,
                 num_samples=10000,
                 sigma=3,
                 epsilon=1,
                 shape=(1, 32, 32)
                 ):

        self.v = v
        self.num_samples = num_samples
        self.sigma = sigma
        self.epsilon = epsilon
        self.shape = shape
        self.data, self.targets = self._generate_dataset(self.num_samples)
        super()

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        return img, target

    def __len__(self):
        return self.num_samples

    def _generate_dataset(self, n_samples):
        if n_samples > 1:
            data_plus = self._generate_samples(n_samples // 2 + n_samples % 2, 0).astype(np.float32)
            labels_plus = np.zeros([n_samples // 2 + n_samples % 2]).astype(np.long)
            data_minus = self._generate_samples(n_samples // 2, 1).astype(np.float32)
            labels_minus = np.ones([n_samples // 2]).astype(np.long)
            data = np.r_[data_plus, data_minus]
            labels = np.r_[labels_plus, labels_minus]
        else:
            data = self._generate_samples(1, 0).astype(np.float32)
            labels = np.zeros([1]).astype(np.long)

        return torch.from_numpy(data), torch.from_numpy(labels)

    def _generate_samples(self, n_samples, label):
        data = self._generate_noise_floor(n_samples)
        sign = 1 if label == 0 else -1
        data = sign * self.epsilon / 2 * self.v[np.newaxis, :] + self._project_orthogonal(data)
        return data

    def _generate_noise_floor(self, n_samples):
        shape = [n_samples] + self.shape
        data = self.sigma * np.random.randn(*shape)

        return data

    def _project(self, x):
        proj_x = np.reshape(x, [x.shape[0], -1]) @ np.reshape(self.v, [-1, 1])
        return proj_x[:, :, np.newaxis, np.newaxis] * self.v[np.newaxis, :]

    def _project_orthogonal(self, x):
        return x - self._project(x)


def generate_synthetic_data(v,
                            num_train=10000,
                            num_test=10000,
                            sigma=3,
                            epsilon=1,
                            shape=(1, 32, 32),
                            batch_size=128):
    trainset = DirectionalLinearDataset(v,
                                        num_samples=num_train,
                                        sigma=sigma,
                                        epsilon=epsilon,
                                        shape=shape)

    testset = DirectionalLinearDataset(v,
                                       num_samples=num_train,
                                       sigma=sigma,
                                       epsilon=epsilon,
                                       shape=shape)

    trainloader = DataLoader(trainset,
                             shuffle=True,
                             pin_memory=True,
                             num_workers=2,
                             batch_size=batch_size)

    testloader = DataLoader(testset,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=2,
                            batch_size=batch_size
                            )

    return trainloader, testloader, trainset, testset


if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    from models import TransformLayer # It normalizes the data to have prespecified mean and stddev
    from models import LeNet

    v = torch.zeros([1, 32, 32])  # Create empty vector
    v_fft = torch.rfft(v, signal_ndim=2)
    v_fft[0, 3, 4, 1] = 1  # Select coordinate in fourier space
    v = torch.irfft(v_fft, signal_ndim=2, signal_sizes=[32, 32])
    v = v / v.norm()
    trainloader, testloader, trainset, testset = generate_synthetic_data(v.numpy(),
                                                                         num_train=10000,
                                                                         num_test=10000,
                                                                         sigma=3,
                                                                         epsilon=1,
                                                                         shape=[1, 32, 32],
                                                                         batch_size=128)

    v = np.random.randn(1, 32, 32)
    v = v / np.linalg.norm(v)
    trainloader, testloader, trainset, testset = generate_synthetic_data(v,
                                                                         num_train=10000,
                                                                         num_test=10000,
                                                                         sigma=3,
                                                                         epsilon=1,
                                                                         shape=[1, 32, 32],
                                                                         batch_size=128)

    # net = LogReg(input_dim=32 * 32, num_classes=2)
    # net = VGG11_bn(num_channels=1, num_classes=2)
    # net = ResNet18(num_channels=1, num_classes=2)
    # net = DenseNet121(num_channels=1, num_classes=2)

    net = LeNet(num_channels=1, num_classes=2)
    net = net.to(DEVICE)

    trained_model = train(model=net,
                          trans=TransformLayer(mean=torch.tensor(0., device=DEVICE),
                                               std=torch.tensor(1., device=DEVICE)),
                          trainloader=trainloader,
                          testloader=testloader,
                          epochs=20,
                          max_lr=0.5,
                          momentum=0,
                          weight_decay=0
                          )