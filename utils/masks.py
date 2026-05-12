import torch
from torch.utils.data import Dataset
from typing import Any

class StaticMaskedDataset(Dataset):
    """
    Wraps a Dataset or Subset to provide deterministic random masks for each image.
    Usage: during validation to ensure consistency across epochs.
    """
    def __init__(self, dataset: Dataset, n_glimpses: int, patch_size: int, experiment_seed: int):
        self.dataset: Any = dataset
        self.n_glimpses = n_glimpses
        self.patch_size = patch_size
        self.seed = experiment_seed

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, target = self.dataset[idx]
        _, H, W = img.shape

        # seed generator using experiment seed + index = deterministic mask per image per experiment
        gen = torch.Generator()
        gen.manual_seed(self.seed + idx)

        mask = torch.zeros((1, H, W), device=img.device)
        for _ in range(self.n_glimpses):
            x = torch.randint(0, W - self.patch_size + 1, (1,), generator=gen).item()
            y = torch.randint(0, H - self.patch_size + 1, (1,), generator=gen).item()
            mask[0, y:y+self.patch_size, x:x+self.patch_size] = 1.0

        return torch.cat([img * mask, mask], dim=0), target

def glimpse_mask(images: torch.Tensor, n_glimpses: int, patch_size: int) -> torch.Tensor:
    """
    Applies n random square patches (with replacement).
    Args:
        images: (N, C, H, W)
        n_glimpses: number of patches to apply
        patch_size: side length of square patch
    Returns:
        masked_images, mask
    """
    N, _, H, W = images.shape
    mask = torch.zeros((N, 1, H, W), device=images.device)

    for _ in range(n_glimpses):
        # Sample top-left corners uniformly
        x = torch.randint(0, W - patch_size + 1, (N,), device=images.device)
        y = torch.randint(0, H - patch_size + 1, (N,), device=images.device)

        for i in range(N):
            mask[i, 0, y[i]:y[i]+patch_size, x[i]:x[i]+patch_size] = 1.0

    return torch.cat([images * mask, mask], dim=1)
    # Sanity check: try this to return noise and a mask. It should just fail of course.
    # return torch.cat([torch.randn_like(images) * mask, mask], dim=1)

def pixelwise_mask(images: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    """
    Applies random pixelwise binary mask over image and adds result as 2nd channel.
    A new mask is generated on each apply.
    Args:
        images (torch.Tensor): Image tensor of shape (N, C, H, W).
        keep_ratio (float): How much of the image is revealed. Determines probability of keeping a pixel.
    Returns:
        torch.Tensor: Masked image with shape (N, C+1, H, W).
    """
    mask = (torch.rand(images.shape, device=images.device) < keep_ratio).float()
    return torch.cat([images * mask, mask], dim=1)

def compute_expected_reveal(H: int = 32, W: int = 32, n_glimpses: int = 10, patch_size: int = 8, trials: int = 1000) -> float:
    """
    Computes empirical E[X(n)], expected percentage of pixels revealed by the glimpse mask.
    """
    dummy_images = torch.zeros((trials, 1, H, W))
    # Note: glimpse_mask returns (images * mask, mask) concatenated
    mask_combined = glimpse_mask(dummy_images, n_glimpses, patch_size)
    # The mask itself is the second channel
    masks = mask_combined[:, 1:2, :, :]
    
    reveal_percentages = masks.mean(dim=(1, 2, 3))
    return float(reveal_percentages.mean().item())

#E_x = compute_expected_reveal(n_glimpses=350, patch_size=8)
#print(f"Empirical E[X(10)]: {E_x:.4f}")
# Result for (32, 32, 8, 10) is 0.4418