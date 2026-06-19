# models/ram.py
"""
Recurrent Attention Model (RAM) — Mnih et al., 2014
Trained with REINFORCE. Uses a single-scale hard-crop glimpse sensor.

Architecture:
    GlimpseNetwork    — fuses image patch features and location features
    LocationNetwork   — predicts the mean (mu_x, mu_y) of the next glimpse location
    BaselineNetwork   — predicts the expected reward for variance reduction
    ActionNetwork     — maps final hidden state to binary classification logit
    RecurrentAttentionModel — ties all components together around a GRUCell core
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sub-networks
# ---------------------------------------------------------------------------

class GlimpseNetwork(nn.Module):
    """
    Converts a raw image patch + its (x,y) location into a single feature vector.

    Design (mirrors the original paper's two-pathway fusion):
        - Patch pathway:    small conv layers -> linear -> hidden_g
        - Location pathway: linear -> hidden_g
        - Fusion: relu(patch_features + loc_features) -> g_t

    Args:
        patch_size:  Side length of the square patch (pixels).
        in_channels: Number of image channels (1 for greyscale).
        hidden_g:    Dimensionality of the fused glimpse feature vector.
    """
    def __init__(self, patch_size: int, in_channels: int = 1, hidden_g: int = 128):
        super().__init__()
        self.patch_size = patch_size

        # Small CNN to encode the patch
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool  = nn.AdaptiveAvgPool2d(4)          # 32 * 4 * 4 = 512

        cnn_out_dim = 32 * 4 * 4
        self.fc_patch = nn.Linear(cnn_out_dim, hidden_g)

        # Linear encoder for the 2-D location
        self.fc_loc = nn.Linear(2, hidden_g)

        self.hidden_g = hidden_g

    def forward(self, patches: torch.Tensor, locations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches:   (N, C, patch_size, patch_size)
            locations: (N, 2) in [-1, 1]
        Returns:
            g_t: (N, hidden_g)
        """
        # Patch pathway
        x = F.relu(self.conv1(patches))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        patch_feat = F.relu(self.fc_patch(x))         # (N, hidden_g)

        # Location pathway
        loc_feat = F.relu(self.fc_loc(locations))     # (N, hidden_g)

        # Element-wise sum then relu (as in the paper)
        return F.relu(patch_feat + loc_feat)           # (N, hidden_g)


class LocationNetwork(nn.Module):
    """
    Maps the current RNN hidden state to the mean of a 2-D Gaussian policy.

    Output is clamped to [-1, 1] via tanh so that locations always map to
    valid image coordinates.

    Args:
        hidden_h: Dimensionality of the RNN hidden state.
        std:      Fixed standard deviation for the Gaussian (exploration parameter).
    """
    def __init__(self, hidden_h: int, std: float = 0.1):
        super().__init__()
        self.fc  = nn.Linear(hidden_h, 2)
        self.std = std

    def forward(self, h_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_t: (N, hidden_h)
        Returns:
            loc:     (N, 2) — sampled location (training) or mean (eval), in [-1, 1]
            log_pi:  (N,)   — log-probability of the chosen location (0 during eval)
        """
        mu = torch.tanh(self.fc(h_t))   # (N, 2)

        if self.training:
            # Sample from Gaussian, clamp to valid range
            dist   = torch.distributions.Normal(mu, self.std)
            loc    = dist.sample().clamp(-1.0, 1.0)            # (N, 2)
            # Sum log-probs over the two independent dimensions (x, y)
            log_pi = dist.log_prob(loc).sum(dim=-1)            # (N,)
        else:
            # Greedy deterministic evaluation: take the mean
            loc    = mu.clamp(-1.0, 1.0)
            log_pi = torch.zeros(mu.size(0), device=mu.device)

        # Detach loc: gradients reach loc_net only through log_pi (REINFORCE)
        return loc.detach(), log_pi


class BaselineNetwork(nn.Module):
    """
    Predicts the expected cumulative reward at each time step.
    Used to reduce variance of the REINFORCE gradient estimate.

    Output is passed through sigmoid to stay in [0, 1] (matching the +1/0 reward range).
    """
    def __init__(self, hidden_h: int):
        super().__init__()
        self.fc = nn.Linear(hidden_h, 1)

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_t: (N, hidden_h)
        Returns:
            b_t: (N,) — predicted baseline value in [0, 1]
        """
        return torch.sigmoid(self.fc(h_t)).squeeze(-1)  # (N,)


class ActionNetwork(nn.Module):
    """
    Produces the final binary classification logit from the last RNN hidden state.
    """
    def __init__(self, hidden_h: int):
        super().__init__()
        self.fc = nn.Linear(hidden_h, 1)

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_t: (N, hidden_h)
        Returns:
            logit: (N, 1) — raw pre-sigmoid classification logit
        """
        return self.fc(h_t)  # (N, 1)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class RecurrentAttentionModelModern(nn.Module):
    """
    Recurrent Attention Model (RAM) — Mnih et al., 2014 (Modern Variant)

    At each of T time steps the model:
        1. Extracts a glimpse at the current location via GlimpseNetwork.
        2. Updates a GRU hidden state.
        3. Predicts a new location (and a baseline for REINFORCE).
    After T steps, ActionNetwork produces the final classification logit.

    Args:
        patch_size:  Side length of the square patch.
        in_channels: Number of image channels (1 for greyscale).
        hidden_g:    Glimpse feature vector dimensionality.
        hidden_h:    GRU hidden state dimensionality.
        std:         Standard deviation for the Gaussian location policy.
    """
    def __init__(
        self,
        patch_size:  int   = 8,
        in_channels: int   = 1,
        hidden_g:    int   = 128,
        hidden_h:    int   = 256,
        std:         float = 0.1,
    ):
        super().__init__()
        self.glimpse_net  = GlimpseNetwork(patch_size, in_channels, hidden_g)
        self.core_rnn     = nn.GRUCell(hidden_g, hidden_h)
            
        self.location_net = LocationNetwork(hidden_h, std)
        self.baseline_net = BaselineNetwork(hidden_h)
        self.action_net   = ActionNetwork(hidden_h)

        self.hidden_h = hidden_h

    def forward(
        self,
        images:       torch.Tensor,
        num_glimpses: int,
        patch_size:   int,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Args:
            images:       (N, C, H, W) — raw, unmasked images.
            num_glimpses: T — number of glimpse steps.
            patch_size:   Side length of the square patch to extract.
        Returns:
            logit:      (N, 1)     — classification logit from the final hidden state.
            log_pis:    [T x (N,)] — log-probs of sampled locations (for REINFORCE loss).
            baselines:  [T x (N,)] — baseline predictions at each step (for variance reduction).
            locations:  [T x (N, 2)] — actual locations chosen at each step.
        """
        from utils.masks import extract_patch  # local import to avoid circular imports

        N      = images.size(0)
        device = images.device

        # Initialise hidden state and first location to the image centre (0, 0 in [-1,1])
        h_t = torch.zeros(N, self.hidden_h, device=device)
        loc = torch.zeros(N, 2, device=device)

        log_pis:   list[torch.Tensor] = []
        baselines: list[torch.Tensor] = []
        locations: list[torch.Tensor] = []

        for _ in range(num_glimpses):
            # 1. Extract patch at current location
            patches = extract_patch(images, loc, patch_size)   # (N, C, p, p)

            # 2. Encode glimpse (patch + location) into feature vector
            g_t = self.glimpse_net(patches, loc)               # (N, hidden_g)

            # 3. Update RNN hidden state
            h_t = self.core_rnn(g_t, h_t)                     # (N, hidden_h)

            # 4. Predict next location and baseline
            loc, log_pi = self.location_net(h_t)               # (N, 2), (N,)
            b_t         = self.baseline_net(h_t)               # (N,)

            log_pis.append(log_pi)
            baselines.append(b_t)
            locations.append(loc)

        # 5. Final classification from last hidden state
        logit = self.action_net(h_t)   # (N, 1)

        return logit, log_pis, baselines, locations
