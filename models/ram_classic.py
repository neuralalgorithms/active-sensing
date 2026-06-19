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

class GlimpseNetworkClassic(nn.Module):
    def __init__(self, patch_size: int, in_channels: int = 1, hidden_g: int = 256):
        super().__init__()
        self.patch_size = patch_size

        # Patch pathway
        self.fc_patch = nn.Linear(patch_size * patch_size * in_channels, 128)
        
        # Location pathway
        self.fc_loc = nn.Linear(2, 128)
        
        # Fusion pathways
        self.fc_g_out = nn.Linear(128, hidden_g)
        self.fc_l_out = nn.Linear(128, hidden_g)

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
        # Paper §4: fl(h) = Linear(h) — no squashing of any kind.
        # Applying tanh after sampling would corrupt log_prob (missing Jacobian),
        # biasing the REINFORCE gradient. The paper relies on training to keep
        # locations in the [-1, 1] coordinate range.
        mu = self.fc(h_t)   # (N, 2)

        if self.training:
            dist   = torch.distributions.Normal(mu, self.std)
            loc    = dist.sample()                        # (N, 2)
            log_pi = dist.log_prob(loc).sum(dim=-1)       # (N,)
        else:
            loc    = mu
            log_pi = torch.zeros(mu.size(0), device=mu.device)

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
        patch_size:  int   = 8,
        in_channels: int   = 1,
        hidden_g:    int   = 256,
        hidden_h:    int   = 256,
        std:         float = 0.1,
    ):
        super().__init__()
        self.glimpse_net  = GlimpseNetworkClassic(patch_size, in_channels, hidden_g)
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
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        from utils.masks import extract_patch

        N      = images.size(0)
        device = images.device

        h_t = torch.zeros(N, self.hidden_h, device=device)
        loc = torch.zeros(N, 2, device=device)

        log_pis:   list[torch.Tensor] = []
        baselines: list[torch.Tensor] = []
        locations: list[torch.Tensor] = []

        for _ in range(num_glimpses):
            patches = extract_patch(images, loc, patch_size)
            g_t = self.glimpse_net(patches, loc)
            h_t = self.core_rnn(g_t, h_t)

            loc, log_pi = self.location_net(h_t)
            b_t         = self.baseline_net(h_t)

            log_pis.append(log_pi)
            baselines.append(b_t)
            locations.append(loc)

        logit = self.action_net(h_t)

        return logit, log_pis, baselines, locations
