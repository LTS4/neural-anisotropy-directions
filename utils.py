import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, trans, trainloader, testloader, epochs, max_lr, momentum, weight_decay):
    lr_schedule = lambda t: np.interp([t], [0, epochs], [max_lr, 0])[0]
    opt = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=momentum, weight_decay=weight_decay)
    loss_fun = nn.CrossEntropyLoss()

    print('Starting training...')
    print()

    best_acc = 0
    for epoch in range(epochs):
        print('Epoch', epoch)
        train_loss_sum = 0
        train_acc_sum = 0
        train_n = 0

        model.train()

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            lr = lr_schedule(epoch + (batch_idx + 1) / len(trainloader))
            opt.param_groups[0].update(lr=lr)

            output = model(trans(inputs))
            loss = loss_fun(output, targets)

            opt.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

            train_loss_sum += loss.item() * targets.size(0)
            train_acc_sum += (output.max(1)[1] == targets).sum().item()
            train_n += targets.size(0)

            if batch_idx % 100 == 0:
                print('Batch idx: %d(%d)\tTrain Acc: %.3f%%\tTrain Loss: %.3f' %
                      (batch_idx, epoch, 100. * train_acc_sum / train_n, train_loss_sum / train_n))

        print('\nTrain Summary\tEpoch: %d | Train Acc: %.3f%% | Train Loss: %.3f' %
              (epoch, 100. * train_acc_sum / train_n, train_loss_sum / train_n))

        test_acc, test_loss = test(model, trans, testloader)
        print('Test  Summary\tEpoch: %d | Test Acc: %.3f%% | Test Loss: %.3f\n' % (epoch, test_acc, test_loss))

    return model


def test(model, trans, testloader):
    loss_fun = nn.CrossEntropyLoss()
    test_loss_sum = 0
    test_acc_sum = 0
    test_n = 0

    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            output = model(trans(inputs))
            loss = loss_fun(output, targets)

            test_loss_sum += loss.item() * targets.size(0)
            test_acc_sum += (output.max(1)[1] == targets).sum().item()
            test_n += targets.size(0)

        test_loss = (test_loss_sum / test_n)
        test_acc = (100 * test_acc_sum / test_n)

        return test_acc, test_loss


def input_numerical_jacobian(fn, x, scale, device, batch_size=None):
    shape = list(x.shape)
    n_dims = int(np.prod(shape))
    batch_size = n_dims if batch_size is None else batch_size
    v = torch.eye(n_dims).view([n_dims] + shape)
    jac = torch.zeros(n_dims)
    residual = 1 if n_dims % batch_size > 0 else 0
    for n in range(n_dims // batch_size + residual):
        batch_plus = x[None, :] + scale * v[n * batch_size: (n+1) * batch_size].to(device)
        batch_minus = x[None, :] - scale * v[n * batch_size: (n+1) * batch_size].to(device)

        jac[n * batch_size: (n+1) * batch_size] = ((fn(batch_plus) - fn(batch_minus)) / (2 * scale)).detach().cpu()[:, 0]

    return jac.view(shape)


def load_cifar_data(path, batch_size=128):
    tf_train = transforms.Compose([transforms.ToTensor()])

    tf_test = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root=path, download=True, train=True, transform=tf_train)
    testset = torchvision.datasets.CIFAR10(root=path, download=True, train=False, transform=tf_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                              num_workers=2, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                             num_workers=2, pin_memory=True)

    mean = torch.as_tensor([0.4914, 0.4822, 0.4465], dtype=torch.float, device=DEVICE)[None, :, None, None]
    std = torch.as_tensor([0.247, 0.243, 0.261], dtype=torch.float, device=DEVICE)[None, :, None, None]

    return trainloader, testloader, trainset, testset, mean, std
