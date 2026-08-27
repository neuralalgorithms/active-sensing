# models/ram_classic.py
"""
Recurrent Attention Model (RAM) — Mnih et al., 2014 (Classic Variant)
Trained with REINFORCE. Uses a single-scale hard-crop glimpse sensor.

Architecture matches the paper:
    GlimpseNetwork    — fully connected pathways (128 units) fused to 256 units
    LocationNetwork   — fl(h) = Linear(h), no squashing (paper §4 verbatim)
                        Coordinates: (0,0) = image centre, (-1,-1) = top-left corner
    BaselineNetwork   — unconstrained linear output trained via MSE on cumulative
                        reward; no sigmoid squashing, per the paper's definition
    ActionNetwork     — maps final hidden state to binary classification logit
    RecurrentAttentionModelClassic — uses Vanilla RNN with ReLU activation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import sample_rejection_gaussian


class GlimpseNetworkClassic(nn.Module):
    def __init__(
        self,
        patch_size: int,
        in_channels: int = 1,
        hidden_patch: int = 128,
        hidden_loc: int = 128,
        hidden_g: int = 256,
    ):
        super().__init__()
        self.patch_size = patch_size

        # Patch pathway
        self.fc_patch = nn.Linear(patch_size * patch_size * in_channels, hidden_patch)
        
        # Location pathway
        self.fc_loc = nn.Linear(2, hidden_loc)
        
        # Fusion pathways
        self.fc_g_out = nn.Linear(hidden_patch, hidden_g)
        self.fc_l_out = nn.Linear(hidden_loc, hidden_g)

    def forward(self, patches: torch.Tensor, locations: torch.Tensor) -> torch.Tensor:
        # Flatten patch
        x = torch.flatten(patches, 1)
        
        h_g = F.relu(self.fc_patch(x))
        h_l = F.relu(self.fc_loc(locations))
        
        g_t = F.relu(self.fc_g_out(h_g) + self.fc_l_out(h_l))
        return g_t


class LocationNetworkClassic(nn.Module):
    def __init__(self, hidden_h: int, std: float = 0.1):
        super().__init__()
        self.fc  = nn.Linear(hidden_h, 2)
        self.std = std

    def forward(self, h_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Bounded mean via tanh squashing: keeps mu strictly within (-1, 1).
        # Does not transform the action variable x itself, preserving Gaussian log-prob
        # without requiring a change-of-variables Jacobian correction.
        mu = torch.tanh(self.fc(h_t))   # (N, 2)

        if self.training:
            loc, log_pi = sample_rejection_gaussian(mu, self.std, low=-1.0, high=1.0)
        else:
            # Deterministic mean (original):
            # loc    = torch.clamp(mu, -1.0, 1.0)
            # log_pi = torch.zeros(mu.size(0), device=mu.device)
            # Sample at inference:
            loc, log_pi = sample_rejection_gaussian(mu, self.std, low=-1.0, high=1.0)

        return loc.detach(), log_pi


class BaselineNetworkClassic(nn.Module):
    def __init__(self, hidden_h: int):
        super().__init__()
        self.fc = nn.Linear(hidden_h, 1)

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        # No sigmoid: the baseline is an unconstrained estimate of the expected
        # cumulative reward, trained via MSE. The paper places no bounds on it.
        return self.fc(h_t).squeeze(-1)


class ActionNetworkClassic(nn.Module):
    def __init__(self, hidden_h: int):
        super().__init__()
        self.fc = nn.Linear(hidden_h, 1)

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        return self.fc(h_t)


class RecurrentAttentionModelClassic(nn.Module):
    def __init__(
        self,
        patch_size:   int = 8,
        in_channels:  int = 1,
        hidden_dim:   int | None = None,
        hidden_patch: int | None = None,
        hidden_loc:   int | None = None,
        hidden_g:     int = 256,
        hidden_h:     int = 256,
        std:        float = 0.1,
    ):
        super().__init__()
        if hidden_dim is not None:
            hidden_g = hidden_dim
            hidden_h = hidden_dim
            hidden_patch = max(2, hidden_dim // 2) if hidden_patch is None else hidden_patch
            hidden_loc = max(2, hidden_dim // 2) if hidden_loc is None else hidden_loc
        else:
            hidden_patch = 128 if hidden_patch is None else hidden_patch
            hidden_loc = 128 if hidden_loc is None else hidden_loc

        self.glimpse_net  = GlimpseNetworkClassic(patch_size, in_channels, hidden_patch, hidden_loc, hidden_g)
        self.core_rnn     = nn.RNNCell(hidden_g, hidden_h, nonlinearity='relu')
            
        self.location_net = LocationNetworkClassic(hidden_h, std)
        self.baseline_net = BaselineNetworkClassic(hidden_h)
        self.action_net   = ActionNetworkClassic(hidden_h)

        self.hidden_h = hidden_h

    def forward(
        self,
        images:       torch.Tensor,
        num_glimpses: int,
        patch_size:   int,
        random_baseline: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        from utils.masks import extract_patch

        N      = images.size(0)
        device = images.device

        h_t = torch.zeros(N, self.hidden_h, device=device)
        loc = torch.zeros(N, 2, device=device)

        log_pis:   list[torch.Tensor] = []
        baselines: list[torch.Tensor] = []
        locations: list[torch.Tensor] = []

        for step in range(num_glimpses):
            # Record the location that will actually be observed this step
            locations.append(loc)

            patches = extract_patch(images, loc, patch_size)
            g_t = self.glimpse_net(patches, loc)
            h_t = self.core_rnn(g_t, h_t)

            # Only sample a next location if there is a subsequent glimpse
            if step < num_glimpses - 1:
                if not random_baseline:
                    loc, log_pi = self.location_net(h_t.detach())
                    # Clamp to [-1, 1] to prevent the coordinate-domain mismatch
                    # from creating an explosive feedback loop into GlimpseNetwork
                    loc = torch.clamp(loc, -1.0, 1.0)
                    b_t = self.baseline_net(h_t.detach())
                else:
                    loc = torch.empty(N, 2, device=device).uniform_(-1.0, 1.0)
                    log_pi = torch.zeros(N, device=device)
                    b_t    = torch.zeros(N, device=device)

                log_pis.append(log_pi)
                baselines.append(b_t)

        logit = self.action_net(h_t)

        return logit, log_pis, baselines, locations
