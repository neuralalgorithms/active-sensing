import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os

# Configuration
FOLDER_NAME = "data/dataset-77-balanced"
CLASSES = {
    "Patchy": "data_patchy.npz",
    "Horizontal": "data_horizontal.npz",
    "Vertical": "data_vertical.npz"
}
SAMPLES_PER_CLASS = 4
N_ROWS = len(CLASSES)
N_COLS = SAMPLES_PER_CLASS

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
})

def visualize():
    fig = plt.figure(figsize=(12, 10))
    
    # Primary Title
    fig.suptitle("Stimuli Dataset (77 x 77 Grid)", fontsize=18, y=0.97)
    
    # Subtitle positioned at 50% width, ~92% height
    fig.text(0.5, 0.93, "Generated via Gaussian Process", 
             fontsize=12, ha='center', va='center', style='italic', alpha=0.8)

    grid = ImageGrid(fig, 111, 
                     nrows_ncols=(N_ROWS, N_COLS),
                     axes_pad=(0.15, 0.4), # type: ignore
                     label_mode="L")

    for row, (label, filename) in enumerate(CLASSES.items()):
        path = os.path.join(FOLDER_NAME, filename)
        with np.load(path) as data:
            images = data["images"]
        indices = np.random.choice(len(images), N_COLS, replace=False)
        
        for col, idx in enumerate(indices):
            ax = grid[row * N_COLS + col]
            ax.imshow(images[idx], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])

            if col == 0:
                ax.set_ylabel(label, fontsize=12, rotation=0, labelpad=40, ha='right', color="#6dcf1d")
            if row == 0:
                ax.set_title(f"Sample {col+1}", color='#e74c3c')

    plt.savefig("dataset_samples_subtitle.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    visualize()
    