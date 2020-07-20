import torch

from models import LogReg
from nad_computation import GradientCovarianceAnisotropyFinder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def model_gen_fun():
    model = LogReg(num_classes=1).eval()
    return model


anisotropy_finder = GradientCovarianceAnisotropyFinder(model_gen_fun=model_gen_fun,
                                                       scale=100,
                                                       num_networks=10000,
                                                       k=1024,
                                                       eval_point=torch.randn([1, 32, 32], device=DEVICE),
                                                       device=DEVICE,
                                                       batch_size=None)

eigenvalues, NADs = anisotropy_finder.estimate_NADs()

np.save('NADs/LogReg_NADs.npy', NADs)
np.save('NADs/LogReg_eigenvals.npy', eigenvalues)