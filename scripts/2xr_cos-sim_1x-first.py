import torch
import torchani
from torchani.datasets import ANIDataset
from torchani.units import hartree2kcalmol
from pathlib import Path
import pandas as pd
import math
from tqdm import tqdm
import time
import numpy as np

device = torch.device('cpu')#'cuda' if torch.cuda.is_available() else 'cpu')
ani2x = torchani.models.ANI2xr().to(device)

ds = torchani.datasets.ANIDataset('../.local/share/torchani/Datasets/ANI-1x-first-conformers.h5')
cos = torch.nn.CosineSimilarity(dim=-1)

df = pd.DataFrame(columns=['Species','Dataset magnitudes', 'Members magnitudes', 'Mean magnitudes', 'Stdev magnitudes', 'Difference', 'Cos Sim', 'Mean Cos Sim'])
forces_df = pd.DataFrame(columns=['Species', 'Dataset forces', 'Members forces', 'Mean forces', 'Stdev forces'])

species_list = []
dataset_magnitudes_list = []
members_magnitudes_list = []
mean_magnitudes_list = []
stdev_magnitudes_list = []
cos_sim_list = []
mean_cos_sim_list = []

ds_forces_list = []
members_forces_list = []
mean_forces_list = []
stdev_forces_list = []

_start = time.perf_counter()

with ds.keep_open('r') as read_ds:
    total_chunks = read_ds.num_chunks(max_size=2500)
    for group, j, conformer in tqdm(read_ds.chunked_items(max_size=2500), total=total_chunks, desc="Processing chunks"):
        species = conformer['species'].to(device)
        coordinates = conformer['coordinates'].to(device)
        ani_input = (species, coordinates)
        
        expected_forces = hartree2kcalmol(conformer['forces'].to(device))
        expected_magnitude = expected_forces.norm(dim=-1)
        
        predicted_forces = hartree2kcalmol(ani2x.members_forces(ani_input).forces)
        mean_forces = predicted_forces.mean(0)
        stdev_forces = predicted_forces.std(0)
        
        members_cos_sim = cos(expected_forces, predicted_forces)
        mean_cos_sim = cos(expected_forces, mean_forces)
        
        t_members_cos_sim = members_cos_sim.transpose(0,1)
        t_predicted_forces = predicted_forces.transpose(0,1)

        predicted_magnitudes = t_predicted_forces.norm(dim=-1)
        stdev_magnitudes = predicted_magnitudes.std(-1)
        
        mean_predicted_magnitude = t_predicted_forces.mean(1).norm(dim=-1)
        
        species_list.extend(species.tolist())
        dataset_magnitudes_list.extend(expected_magnitude.tolist())
        members_magnitudes_list.extend(predicted_magnitudes.tolist())
        mean_magnitudes_list.extend(mean_predicted_magnitude.tolist())
        stdev_magnitudes_list.extend(stdev_magnitudes.tolist())
        cos_sim_list.extend(t_members_cos_sim.tolist())
        mean_cos_sim_list.extend(mean_cos_sim.tolist())
            
        ds_forces_list.extend(expected_forces.tolist())
        members_forces_list.extend(t_predicted_forces.tolist())
        mean_forces_list.extend(mean_forces.tolist())
        stdev_forces_list.extend(stdev_forces.tolist())

    
df['Species'] = species_list
df['Dataset magnitudes'] = dataset_magnitudes_list
df['Members magnitudes'] = members_magnitudes_list
df['Mean magnitudes'] = mean_magnitudes_list
df['Stdev magnitudes'] = stdev_magnitudes_list
df['Difference'] = df.apply(lambda row: [abs(a - b) for a, b in zip(row['Dataset magnitudes'], row['Mean magnitudes'])], axis=1)
df['Cos Sim'] = cos_sim_list
df['Mean Cos Sim'] = mean_cos_sim_list

forces_df['Species'] = species_list
forces_df['Dataset forces'] = ds_forces_list
forces_df['Members forces'] = members_forces_list
forces_df['Mean forces'] = mean_forces_list
forces_df['Stdev forces'] = stdev_forces_list
forces_df['Cos Sim'] = cos_sim_list
forces_df['Mean Cos Sim'] = mean_cos_sim_list



print(df)
print(forces_df)

df = df.set_index('Species')
forces_df = forces_df.set_index('Species')
df.to_parquet('2xr_cos_sim-mag_df-1x-first.pq')
forces_df.to_csv('2xr_cos_sim-forces_df-1x-first.csv')

print(f"Time elapsed: {time.perf_counter() - _start} s")
