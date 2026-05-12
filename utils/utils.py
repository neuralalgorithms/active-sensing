# utils.py
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv() # Load env vars from .env if exists

def get_dataloaders(grid_size: int=32, batch_size: int=32, seed: int=42, split: float=0.2) -> tuple[DataLoader, DataLoader]: # you could go further with tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], DataLoader[tuple[torch.Tensor, torch.Tensor]]], but there's a limit to how much type hinting one should do! 
    """
    Creates balanced dataloaders for binary classification of patchy vs. stripy grids.
    """

    # look for SLURM_TMPDIR; if not, look for DATASET_ROOT from .env; if not, create local folder "data"
    default_data_root = os.getenv("DATASET_ROOT", "data")
    base_data_dir = os.getenv("SLURM_TMPDIR", default_data_root)

    path = Path(base_data_dir) / f"dataset-{grid_size}-balanced"

    if not path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {path.resolve()}")
    
    try:
        data_p = np.load(path / "data_patchy.npz")["images"]
        data_h = np.load(path / "data_horizontal.npz")["images"]
        data_v = np.load(path / "data_vertical.npz")["images"]
    except Exception as e:
        raise IOError(f"Error: Could not load .npz files from {path}. Check names/format. \n{e}")
    
    X_stripy = np.concatenate([data_h, data_v], axis=0) # Combine horizontal and vertical into a single class "stripy" (class 0)

    X = np.concatenate([data_p, X_stripy], axis=0)
    y = np.array([1] * len(data_p) + [0] * len(X_stripy))

    # Add Channel dimension (1) for CNN
    X_tensor = torch.from_numpy(X).unsqueeze(1)
    y_tensor = torch.from_numpy(y).float().unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)

    # Split
    train_len = int((1 - split) * len(dataset))
    test_len = len(dataset) - train_len

    num_workers = int(os.getenv("SLURM_CPUS_PER_TASK", 0)) # use multiple CPUs if available

    train_ds, test_ds = random_split(
        dataset, [train_len, test_len], generator=torch.Generator().manual_seed(seed)
    )

    return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers), \
           DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def save_to_csv(filename, data):
    """
    Saves a list of dictionaries to a CSV. 
    Appends if the file exists, creates it with headers if it doesn't.
    """
    # look for OUTPUT_DIR from .env; if not, create local folder "output"
    output_base = Path(os.getenv("OUTPUT_DIR", "./output")) # ./output = output
    full_path = output_base / filename

    full_path.parent.mkdir(parents=True, exist_ok=True) # Check if parent directory exists. Analogous to os.makedirs(os.path.dirname(filename), exist_ok=True)

    df = pd.DataFrame(data)

    if full_path.exists(): # If file exists
        df.to_csv(full_path, mode='a', index=False, header=False) # Append without header
    else: # If file does not exist
         df.to_csv(full_path, index=False) # Write with header
    # A cool one-liner version: df.to_csv(full_path, mode='a', index=False, header=not full_path.exists())

def load_weights(model, path):
    if not os.path.exists(path):
        return False
    try:
        model.load_state_dict(torch.load(path))
        return True
    except FileNotFoundError:
        print(f"Error: No weights found at {path}.")
        return False

def save_weights(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

# --- LEGACY CODE ---

def get_dataloaders_unbalanced(batch_size=32, seed=42):
    # Load raw data
    data_p = np.load("data/dataset-32/data_patchy.npz")
    data_h = np.load("data/dataset-32/data_horizontal.npz")
    data_v = np.load("data/dataset-32/data_vertical.npz")

    X_raw = np.vstack([data_p["images"], data_h["images"], data_v["images"]])

    # Patchy = 1, Stripy (Horizontal + Vertical) = 0
    y = np.concatenate(
        [
            np.ones_like(data_p["labels"]),
            np.zeros_like(data_h["labels"]),
            np.zeros_like(data_v["labels"]),
        ]
    ).astype(np.int64)  # Cast integer

    if X_raw.ndim == 3:
        X_raw = X_raw[:, np.newaxis, :, :]

    dataset = TensorDataset(
        torch.tensor(X_raw, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(
            1
        ),  # Ig can unsqueeze here- changes from [N] to [N, 1]
    )

    # Split with a fixed seed for consistency across scripts
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_dataloaders_corrector(batch_size=32, seed=42):
    # Load raw data
    data_p = np.load("data/dataset-32/data_patchy.npz")
    data_h = np.load("data/dataset-32/data_horizontal.npz")
    data_v = np.load("data/dataset-32/data_vertical.npz")

    # Balancing: Match Stripy count to Patchy count
    n_patchy = len(data_p["images"])
    n_per_stripy = n_patchy // 2

    X_patchy = data_p["images"]
    X_stripy = np.vstack(
        [data_h["images"][:n_per_stripy], data_v["images"][:n_per_stripy]]
    )

    X_raw = np.vstack([X_patchy, X_stripy])
    y = np.concatenate([np.ones(len(X_patchy)), np.zeros(len(X_stripy))]).astype(
        np.float32
    )

    if X_raw.ndim == 3:
        X_raw = X_raw[:, np.newaxis, :, :]

    dataset = TensorDataset(torch.tensor(X_raw), torch.tensor(y).unsqueeze(1))

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed)
    )

    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(
        test_ds, batch_size=batch_size, shuffle=False
    )
