import numpy as np
import scipy.spatial.distance as dist
import time
import matplotlib.pyplot as plt
import sys
from pathlib import Path

SPAN: float=27.8
GRID_SIZE: int=77
CLASS_MAP: dict[str, int] = {"patchy": 0, "horizontal": 1, "vertical": 2}


def k_patchy(xa: np.ndarray, xb: np.ndarray, length_scale: float = 1.39) -> np.ndarray:
    """Paper-accurate isotropic RBF kernel for patches/blobs (Yang et al.)."""
    sq_dist = dist.cdist(xa, xb, "sqeuclidean")
    phi = sq_dist / (length_scale**2)
    return np.exp(-0.5 * phi)


def k_anisotropic(xa: np.ndarray, xb: np.ndarray, lh: float, lv: float) -> np.ndarray:
    """Paper-accurate anisotropic RBF kernel for horizontal/vertical stripes (Yang et al.)."""
    u_dist_sq = dist.cdist(xa[:, :1], xb[:, :1], "sqeuclidean")
    v_dist_sq = dist.cdist(xa[:, 1:], xb[:, 1:], "sqeuclidean")
    phi = (u_dist_sq / (lh**2)) + (v_dist_sq / (lv**2))
    return np.exp(-0.5 * phi)


def k_patchy_inverse(xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
    """Legacy isotropic RBF kernel (treats 1.39 as precision/frequency)."""
    sq_dist = dist.cdist(xa, xb, "sqeuclidean")
    phi = (1.39**2) * sq_dist
    return np.exp(-0.5 * phi)


def k_anisotropic_inverse(xa: np.ndarray, xb: np.ndarray, lh: float, lv: float) -> np.ndarray:
    """Legacy anisotropic RBF kernel (treats lh, lv as precision/frequency)."""
    u_dist_sq = dist.cdist(xa[:, :1], xb[:, :1], "sqeuclidean")
    v_dist_sq = dist.cdist(xa[:, 1:], xb[:, 1:], "sqeuclidean")
    phi = (lh**2 * u_dist_sq) + (lv**2 * v_dist_sq)
    return np.exp(-0.5 * phi)


def generate_valid_sample(L: np.ndarray, num_vars: int) -> np.ndarray:
    """Generates a sample and rejects samples outside bounds [-4, 4] """
    while True:
        z = np.random.standard_normal(num_vars)  # generate N^2 uncorrelated points
        f = L @ z  # make them correlated according to this covariance matrix
        if np.all(f >= -4) and np.all(f <= 4): # Rejection sampling
            return f


def get_cholesky(X: np.ndarray, class_name: str) -> np.ndarray:
    """Computes Cholesky decomp of covariance matrix for a given class."""
    print(f"Computing {GRID_SIZE} x {GRID_SIZE} Covariance Matrix and Cholesky Decomp for {class_name}...")

    if class_name == "patchy":
        covar = k_patchy(X, X, length_scale=1.39)
    elif class_name == "vertical":
        covar = k_anisotropic(X, X, lh=0.91, lv=4.63)
    elif class_name == "horizontal":
        covar = k_anisotropic(X, X, lh=4.63, lv=0.91)
    else:
        raise ValueError(f"Unknown class: {class_name}")
    covar += 1e-6 * np.eye(len(X))
    return np.linalg.cholesky(covar)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <class> <num_samples>")
        sys.exit(1)

    image_class = sys.argv[1]
    num_samples = int(sys.argv[2])

    start = time.time()

    axis = np.linspace(0, SPAN, GRID_SIZE)
    u, v = np.meshgrid(axis, axis)
    X = np.stack([u.ravel(), v.ravel()], axis=1)

    L = get_cholesky(X, image_class)
    X_data = np.zeros((num_samples, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    y_labels = np.full((num_samples,), CLASS_MAP[image_class], dtype=np.int64)

    print(f"Generating {num_samples} samples for '{image_class}'...")

    for i in range(num_samples):
        if i % 100 == 0 and i > 0:
            print(f"  ...generated {i}/{num_samples}")
        raw_f = generate_valid_sample(L, len(X))

        normalized_data = (raw_f + 4) / 8  # Map to [0.0, 1.0] for greyscale
        X_data[i] = normalized_data.reshape(GRID_SIZE, GRID_SIZE)

        # --- visual approval ---
        if i == 0:
            print(
                "Visualizing the first sample. Close the image window to continue generating the rest..."
            )
            plt.imshow(X_data[i], cmap="gray", vmin=0, vmax=1)
            plt.title(f"Class: {image_class} - Close to continue")
            plt.axis("off")  # Hides the axes for a cleaner look
            plt.show()  # The script pauses here until you close the window
            print("Resuming generation...")
        # --------------------------------

    filepath = Path(f"data/test-77-balanced/data_{image_class}.npz")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(filepath, images=X_data, labels=y_labels)

    print(
        f"Generation complete. Saved to {filepath} in {time.time() - start:.2f} seconds."
    )
