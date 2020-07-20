import numpy as np

import torch

def poison_with_NADs(trainset, NAD_idx, epsilon, NAD_path, num_classes=10, num_channels=3, batch_size=128):
    x = torch.from_numpy(trainset.data.transpose([0, 3, 1, 2])).type(torch.float) / 255.
    y = torch.tensor(trainset.targets, dtype=torch.long)
    shape = x.shape[1:]

    V = np.load(NAD_path)
    V = torch.from_numpy(V)
    poison_indices = (NAD_idx, NAD_idx + 1)

    x_poison = x.clone()
    for t in range(num_classes):
        idx = poison_indices[t // (num_channels * 2)]
        channel_idx = t % num_channels
        sign = 2 * (t % 2) - 1
        carrier = torch.zeros_like(x[0])
        carrier[channel_idx] = V[idx].view([1, shape[-2], shape[-1]])
        x_bias = torch.einsum('bi, i->b', x[y == t].view([-1, np.prod(shape)]), carrier.view(-1))
        x_poison[y == t] += (epsilon * sign - x_bias[:, None, None, None]) * carrier[None, :, :, :]

    poisonset = torch.utils.data.TensorDataset(x_poison, y)
    poisonloader = torch.utils.data.DataLoader(poisonset, batch_size=batch_size, shuffle=True, num_workers=2,
                                               pin_memory=True)

    return poisonloader


if __name__ == '__main__':
    from models import ResNet18, TransformLayer
    from utils import train, load_cifar_data

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    architecture = 'ResNet18'
    net = ResNet18(num_channels=3, num_classes=10)
    net = net.to(DEVICE)

    CIFAR_path = './'
    NAD_dir = './NADs/'
    NAD_path = NAD_dir + architecture + '_NADs.npy'

    poison_idx = 0
    epsilon = 0.05

    trainloader, testloader, trainset, testset, mean, std = load_cifar_data(CIFAR_path)

    poisonloader = poison_with_NADs(trainset,
                                    NAD_idx=poison_idx,
                                    epsilon=epsilon,
                                    NAD_path=NAD_path)

    trained_model = train(model=net,
                          trans=TransformLayer(mean=mean, std=std),
                          trainloader=poisonloader,
                          testloader=testloader,
                          epochs=50,
                          max_lr=0.21,
                          momentum=0.9,
                          weight_decay=5e-4
                          )
