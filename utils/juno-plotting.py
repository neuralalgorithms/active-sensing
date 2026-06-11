import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
from torch.utils.data import TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from masks import StaticMaskedDataset

# 1. Define the path pattern to find all your files
current_dir = os.path.dirname(os.path.abspath(__file__))
cached_csv_path = os.path.join(current_dir, "combined_with_coverage.csv")

if os.path.exists(cached_csv_path):
    print("Loading cached dataframe...")
    combined_df = pd.read_csv(cached_csv_path)
    
    # We still need unique_patch_sizes later
    df_runs = combined_df.groupby(['patch_size', 'glimpses', 'seed']).agg({
        'val_accuracy': 'max',
        'val_loss': 'min',
        'actual_coverage': 'first',
        'theoretical_coverage': 'first'
    }).reset_index()
    
else:
    file_pattern = os.path.join(current_dir, "..", "from-juno", "moderatelysmall_p8", "results_seed_*.csv")
    file_list = glob.glob(file_pattern)
    
    print(f"Found {len(file_list)} files to combine.")
    
    # 2. Loop through and read each file, appending the seed as a column
    dataframes = []
    for filepath in file_list:
        df = pd.read_csv(filepath)
        
        # Extract the seed number from the filename (e.g., 'results_seed_12.csv' -> 12)
        filename = os.path.basename(filepath)
        seed_number = int(filename.split('_')[-1].split('.')[0])
        
        # Add the seed column to the DataFrame
        df['seed'] = seed_number
        
        dataframes.append(df)
    
    # 3. Concatenate all DataFrames into one single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    H, W = 77, 77
    VAL_SET_SIZE = 800
    
    def compute_actual_mean_coverage(H, W, n_glimpses, patch_size, seed, val_set_size):
        dummy_data = torch.zeros((val_set_size, 1, H, W))
        dummy_targets = torch.zeros(val_set_size)
        dummy_dataset = TensorDataset(dummy_data, dummy_targets)
        
        static_ds = StaticMaskedDataset(dummy_dataset, n_glimpses, patch_size, seed)
        
        total_coverage = 0.0
        for idx in range(val_set_size):
            img_with_mask, _ = static_ds[idx]
            mask = img_with_mask[1]  # The mask is the second channel
            total_coverage += mask.mean().item()
            
        return total_coverage / val_set_size
    
    unique_configs = combined_df[['seed', 'glimpses', 'patch_size']].drop_duplicates()
    actual_cov_map = {}
    
    print("Computing coverage metrics (this might take a minute)...")
    for _, row in unique_configs.iterrows():
        seed = int(row['seed'])
        glimpses = int(row['glimpses'])
        patch_size = int(row['patch_size'])
        
        cov = compute_actual_mean_coverage(H, W, glimpses, patch_size, seed, VAL_SET_SIZE)
        actual_cov_map[(seed, glimpses, patch_size)] = cov
    
    combined_df['actual_coverage'] = combined_df.apply(
        lambda r: actual_cov_map[(int(r['seed']), int(r['glimpses']), int(r['patch_size']))], 
        axis=1
    )
    
    # Calculate Theoretical Coverage analytically: 1 - (1 - p)^n
    # where p is the ratio of patch area to total image area
    combined_df['theoretical_coverage'] = combined_df.apply(
        lambda r: 1 - (1 - (r['patch_size']**2) / (H * W))**r['glimpses'], 
        axis=1
    )
    
    print(f"Saving dataframe to {cached_csv_path}")
    combined_df.to_csv(cached_csv_path, index=False)
    
    # Grouping for accuracy and loss by seed
    df_runs = combined_df.groupby(['patch_size', 'glimpses', 'seed']).agg({
        'val_accuracy': 'max',
        'val_loss': 'min',
        'actual_coverage': 'first',
        'theoretical_coverage': 'first'
    }).reset_index()

# Calculate Mean, STD, and Count across seeds for each glimpse count
stats = df_runs.groupby(['patch_size', 'glimpses']).agg({
    'val_accuracy': ['mean', 'std', 'count'],
    'val_loss': ['mean', 'std', 'count'],
    'actual_coverage': ['mean'],
    'theoretical_coverage': ['mean']
}).reset_index()

# Filter out patches
# stats = stats[stats['patch_size'] != 3]

# Flatten the multi-index columns for easier plotting access
stats.columns = ['_'.join(col).strip('_') for col in stats.columns]

# Calculate SEM (Standard Error of the Mean)
stats['val_accuracy_sem'] = stats['val_accuracy_std'] / np.sqrt(stats['val_accuracy_count'])
stats['val_loss_sem'] = stats['val_loss_std'] / np.sqrt(stats['val_loss_count'])

# Create a 2x4 grid of subplots with a generous figure size
fig, axes = plt.subplots(2, 4, figsize=(24, 14))

# Flatten the axes array from 2x4 to a 1D array of 8 elements for easier indexing
ax_list = axes.flatten()

unique_patch_sizes = sorted(stats['patch_size'].unique())

# --- Plot 1: Glimpse Count vs. Best Validation Accuracy (Top-Left) ---
ax = ax_list[0]
for ps in unique_patch_sizes:
    subset = stats[stats['patch_size'] == ps]
    ax.errorbar(
        subset['glimpses'], subset['val_accuracy_mean'], 
        yerr=subset['val_accuracy_std'], 
        label=f"Patch Size: {ps}", capsize=5, marker='o'
    )
ax.set_title("Glimpse Count vs. Best Validation Accuracy", fontsize=12, fontweight='bold')
ax.set_xlabel("Number of Glimpses")
ax.set_ylabel(r"Accuracy ($mean \pm std$)")
ax.grid(True, linestyle='--', alpha=0.6)

# --- Plot 2: Glimpse Count vs. Minimum Validation Loss (Top-Right) ---
ax = ax_list[1]
for ps in unique_patch_sizes:
    subset = stats[stats['patch_size'] == ps]
    ax.errorbar(
        subset['glimpses'], subset['val_loss_mean'], 
        yerr=subset['val_loss_std'], 
        label=f"Patch Size: {ps}", capsize=5, marker='s', linestyle='--'
    )
ax.set_title("Glimpse Count vs. Minimum Validation Loss", fontsize=12, fontweight='bold')
ax.set_xlabel("Number of Glimpses")
ax.set_ylabel(r"Min Loss ($mean \pm std$)")
ax.grid(True, linestyle='--', alpha=0.6)

# --- Plot 3: Glimpse Count vs Standard Error of Mean Best Val Accuracy (Bottom-Left) ---
ax = ax_list[2]
for ps in unique_patch_sizes:
    subset = stats[stats['patch_size'] == ps]
    ax.errorbar(
        subset['glimpses'], 
        subset['val_accuracy_mean'], 
        yerr=subset['val_accuracy_sem'], 
        label=f"Patch Size: {ps}", 
        capsize=5, 
        marker='o'
    )
ax.set_title("SE of Mean Best Val Accuracy", fontsize=12, fontweight='bold')
ax.set_xlabel("Number of Glimpses")
ax.set_ylabel(r"Accuracy ($mean \pm SEM$)")
ax.grid(True, linestyle='--', alpha=0.6)

# --- Plot 4: Glimpse Count vs Standard Error of Mean Min Val Loss (Bottom-Right) ---
ax = ax_list[3]
for ps in unique_patch_sizes:
    subset = stats[stats['patch_size'] == ps]
    ax.errorbar(
        subset['glimpses'], 
        subset['val_loss_mean'], 
        yerr=subset['val_loss_sem'], 
        label=f"Patch Size: {ps}", 
        capsize=5, 
        marker='o'
    )
ax.set_title("SE of Mean Minimum Val Loss", fontsize=12, fontweight='bold')
ax.set_xlabel("Number of Glimpses")
ax.set_ylabel(r"Loss ($mean \pm SEM$)")
ax.grid(True, linestyle='--', alpha=0.6)

# --- Plot 5: Validation Accuracy vs Epochs ---
ax = ax_list[4]
df_epochs = combined_df.groupby(['patch_size', 'glimpses', 'epoch']).agg({'val_accuracy': ['mean', 'std']}).reset_index()
df_epochs.columns = ['_'.join(col).strip('_') for col in df_epochs.columns]

for (ps, gl), group in df_epochs.groupby(['patch_size', 'glimpses']):
    ax.plot(group['epoch'], group['val_accuracy_mean'], label=f'PS:{ps}, GL:{gl}')
ax.set_title("Validation Accuracy vs Epochs", fontsize=12, fontweight='bold')
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Accuracy (Mean)")
ax.grid(True, linestyle='--', alpha=0.6)
# Add a small legend specifically for this plot since it has different lines
ax.legend(fontsize=8, loc='best')

# --- Plot 6: Accuracy vs Theoretical Coverage ---
ax = ax_list[5]
for ps in unique_patch_sizes:
    subset = stats[stats['patch_size'] == ps]
    ax.plot(subset['theoretical_coverage_mean'], subset['val_accuracy_mean'], marker='o', label=f"Patch Size {ps}")
ax.set_title("Accuracy vs Theoretical Coverage", fontsize=12, fontweight='bold')
ax.set_xlabel("Theoretical Image Coverage")
ax.set_ylabel("Max Validation Accuracy")
ax.grid(True, linestyle='--', alpha=0.6)

# --- Plot 7: Accuracy vs Actual Coverage ---
ax = ax_list[6]
for ps in unique_patch_sizes:
    subset = stats[stats['patch_size'] == ps]
    ax.plot(subset['actual_coverage_mean'], subset['val_accuracy_mean'], marker='s', linestyle='--', label=f"Patch Size {ps}")
ax.set_title("Accuracy vs Actual Coverage", fontsize=12, fontweight='bold')
ax.set_xlabel("Actual Mean Image Coverage")
ax.set_ylabel("Max Validation Accuracy")
ax.grid(True, linestyle='--', alpha=0.6)

# --- Hide Plot 8 ---
ax_list[7].axis('off')

# --- Global Legend Layout ---

# 1. Extract handles and labels from the first axes (since they are identical for all)
handles, labels = ax_list[0].get_legend_handles_labels()

# 2. Create a single figure-level legend at the bottom
fig.legend(
    handles, labels, 
    loc='upper left', 
    ncol=len(unique_patch_sizes), 
    fontsize=11,
    title_fontsize=12
)

# 3. Adjust layout to prevent the legend from clipping or overlapping the plots
plt.tight_layout()
fig.subplots_adjust(bottom=0.08)  # Creates space at the bottom specifically for the legend

plt.show()