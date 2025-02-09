import time
from tqdm import tqdm

import torch

from torchani.datasets import COMP6v1
import torchani


ds = COMP6v1()

model = torchani.models.ANI2xr()

total_predictions = 0

_start = time.perf_counter()
with ds.keep_open("r") as rds:
    total_chunks = rds.num_chunks(max_size=2500)
    for group, j, conformer in tqdm(rds.chunked_items(max_size=2500), total=total_chunks, desc="Processing chunks:"):
        species = conformer["species"]
        coordinates = conformer["coordinates"]
        coordinates.requires_grad_(True)
        energies = model((species, coordinates)).energies
        print("energy shape:", energies.shape)
        total_predictions += len(energies)
time_elapsed = time.perf_counter()-_start
print(f"Time elapsed: {time_elapsed} s")
print(f"Total predictions: {total_predictions}")
print(f"Predictions per second: {total_predictions / time_elapsed}")
print(f"Inverse of that (time per prediction): {time_elapsed / total_predictions} seconds")

# This took about 7.3 min in CPU of my 2017 MacBook Pro
# Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz

# On a 2024 M4 Pro, it took 39.26 seconds.

