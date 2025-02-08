import torchani
import torch
from torch import Tensor

from tqdm.auto import tqdm

ds = torchani.datasets.ANIDataset('/home/nick/ANI-1x-first-conformers.h5')


def cov(tensor: Tensor, unbiased: bool = True) -> Tensor:
    tensor = tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(unbiased))
    return factor * tensor @ tensor.transpose(-1, -2).conj()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchani.models.ANI2x(periodic_table_index=True).to(device).double()
print(device)
print(model)

for k, v in tqdm(ds.items()):
    c = v["coordinates"][0].unsqueeze(0).to(torch.double)
    s = v["species"][0].unsqueeze(0)

    energies = model.atomic_energies((s, c), average=False).energies.squeeze(1)
    cov_matrix = cov(energies)
    triu = torch.triu_indices(row=energies.shape[-1], col=energies.shape[-1])
    variance = cov_matrix.view(-1).sum()

    member_energies = model.members_energies((s, c)).energies
    variance_expect = torch.var(member_energies, unbiased=True)
    assert torch.isclose(variance, variance_expect)

def cov(atomic_energies):
    """Compute the covariance for a tensor of size (M,N_atoms) 
    where M is the number of models and N_atoms is the number of atoms in the system"""
    M, N_atoms = atomic_energies.shape
    covariance = np.zeros((N_atoms, N_atoms))
    mean = np.mean(atomic_energies, axis=0)
    
    for i in range(N_atoms):
        for j in range(N_atoms):
            covariance[i,j] += (atomic_energies[:,i] - mean[i]) @ (atomic_energies[:,j] - mean[j])

    return covariance
