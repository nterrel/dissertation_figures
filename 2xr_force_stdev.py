import torch
import torchani
from torchani.datasets import COMP6v1
import pandas as pd
from tqdm import tqdm

device = torch.device('cpu')
print('Device:', device)

model = torchani.models.ANI2xr().to(device)
model.set_enabled('energy_shifter', False)

ds = torchani.datasets.ANIDataset('ANI-1x-first-conformers.h5')

results = []
loops = 0

with ds.keep_open("r") as read_ds:
    total_chunks = read_ds.num_chunks(max_size=2500)
    for group, j, conformer in tqdm(read_ds.chunked_items(max_size=2500), total=total_chunks, desc="Processing chunks"):
        species = conformer["species"].to(device)
        coordinates = conformer["coordinates"].to(device)
        dataset_energies = conformer['energies']
        dataset_forces = conformer['forces']
        dataset_energies_cpu = dataset_energies.detach().cpu()
        dataset_forces_cpu = dataset_forces.detach().cpu()

        ani_input = (species, coordinates)
        model.set_enabled('energy_shifter', True)
        e_qbc = model.energies_qbcs(ani_input)
        energies_mean = e_qbc.energies.detach().cpu()
        energies_qbc = e_qbc.qbcs.detach().cpu()

        force_qbc = model.force_qbc(ani_input, ensemble_values=True)
        magnitudes = force_qbc.magnitudes.detach().cpu()
        species_cpu = species.detach().cpu()

        # Compute per-atom standard deviation across ensemble predictions per structure
        stdev_per_structure = magnitudes.std(dim=0).tolist()

        n_structures = energies_mean.shape[0]

        for i in range(n_structures):
            struct_id = f"{group}-{j}-idx{i}"
            species_list = species_cpu[i].tolist()
            dft_energies = dataset_energies_cpu[i].item()
            dft_forces_2d = dataset_forces_cpu[i]
            dft_force_mags = dft_forces_2d.norm(dim=1)
            dft_force_mags_list = dft_force_mags.tolist()
            energy_this_structure = energies_mean[i].tolist()
            energy_qbc_this_structure = energies_qbc[i].tolist()
            force_mags_list = magnitudes[:, i, :].tolist()
            stdev_this_structure = stdev_per_structure[i]  # Ensure correct indexing
            mean_rel_stdev = torch.tensor(stdev_this_structure).mean().item()  # Compute per-structure mean

            record = {
                "id": struct_id,
                "species": species_list,
                "dataset_energy": dft_energies,
                "dataset_force_magnitudes": dft_force_mags_list,
                "energy_mean": energy_this_structure,
                "energy_qbc": energy_qbc_this_structure,
                "force_magnitudes": force_mags_list,
                "force_stdev": stdev_this_structure,
                "mean_stdev": mean_rel_stdev,  # Ensure per-structure mean is used
            }

            results.append(record)
            loops += 1
            print(f"{loops} loops completed, struct_id: {struct_id}")

# Convert results into a DataFrame
df = pd.DataFrame(results)

# Write to parquet
df.to_parquet("2xr_1x-first-isolator.pq", index=False)

print("Done! Results stored in df and written to 2xr_1x-first-isolator.pq")

