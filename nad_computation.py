import numpy as np

import torch
from sklearn.decomposition import PCA

from utils import input_numerical_jacobian


class GradientCovarianceAnisotropyFinder:

    def __init__(self,
                 model_gen_fun,
                 num_networks,
                 eval_point=None,
                 k=None,
                 scale=1,
                 device='cpu',
                 batch_size=None):

        self.model_gen_fun = model_gen_fun
        self.num_networks = num_networks
        self.eval_point = eval_point
        self.scale = scale
        self.k = k
        self.device = device
        self.batch_size = batch_size
        self._gradients = None


    def _numerical_input_derivative(self, model, v0):
        model = model.to(self.device)
        fn = lambda x: -model(x)
        jac = input_numerical_jacobian(fn, v0, self.scale, self.device, batch_size=self.batch_size)
        return jac

    @property
    def sample_gradients(self):
        if self._gradients is None:
            self._gradients = []
            for n in range(self.num_networks):
                self._gradients.append(self._numerical_input_derivative(self.model_gen_fun(), self.eval_point).cpu().view([-1]))
        return torch.stack(self._gradients).numpy()

    def estimate_NADs(self):
        pca = PCA(n_components=self.k)
        pca.fit(self.sample_gradients)
        return pca.singular_values_, pca.components_


if __name__ == '__main__':
    import seaborn as sns
    import matplotlib.pyplot as plt
    from models import LeNet

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


    def model_gen_fun():
        model = LeNet(num_classes=1, num_channels=1).eval()
        return model


    anisotropy_finder = GradientCovarianceAnisotropyFinder(model_gen_fun=model_gen_fun,
                                                           scale=100,
                                                           num_networks=10000,
                                                           k=1024,
                                                           eval_point=torch.randn([1, 32, 32], device=DEVICE),
                                                           device=DEVICE,
                                                           batch_size=None)

    eigenvalues, NADs = anisotropy_finder.estimate_NADs()

    indices = list(range(5))

    plt.figure(figsize=(15, 5))

    for n, index in enumerate(indices):
        x = NADs[index].reshape([32, 32])

        vmax = np.max([np.abs(x.max()), np.abs(x.min())])
        vmin = -vmax

        cmap = sns.cubehelix_palette(8, start=.5, rot=-.75, as_cmap=True, reverse=True)

        x_fft = np.fft.fftshift(np.fft.fft2(x))

        plt.subplot(2 * np.ceil(len(indices) / 5), 5, n + 5 * (n // 5) + 1)
        plt.imshow(x, cmap='BrBG', vmin=vmin, vmax=vmax)
        plt.title(r'Index %d' % index)
        plt.axis('off')

        plt.subplot(2 * np.ceil(len(indices) / 5), 5, n + 5 * (n // 5 + 1) + 1)
        plt.imshow(np.abs(x_fft) ** 2, cmap=cmap)
        plt.axis('off')

    plt.savefig('nads_lenet.pdf')