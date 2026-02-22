"""
StyleNet: a small residual CNN that predicts which move a player would make
given a board position.

Input:  (batch, 18, 8, 8)  — board encoded as binary planes
Output: (batch, 4096)      — logits over all possible (from, to) move pairs

Training objective: cross-entropy against the move the player actually played.
At inference, we mask illegal moves and softmax over the legal subset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from style.features import NUM_PLANES, MOVE_VOCAB


class ResBlock(nn.Module):
    """Standard residual block: two 3×3 convs with a skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class StyleNet(nn.Module):
    """
    Convolutional policy network for player style imitation.

    Architecture:
        Input conv  → ResBlock × num_res_blocks → policy head → 4096 logits

    Args:
        channels:       Number of feature maps in the residual tower.
        num_res_blocks: Depth of the residual tower.
    """

    def __init__(self, channels: int = 128, num_res_blocks: int = 6):
        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        self.res_tower = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_res_blocks)]
        )

        # Policy head: compress spatial dims then project to move vocabulary
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, MOVE_VOCAB),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, NUM_PLANES, 8, 8)

        Returns:
            logits: (batch, MOVE_VOCAB)  — raw, unmasked
        """
        x = self.input_conv(x)
        x = self.res_tower(x)
        return self.policy_head(x)

    def move_probs(
        self,
        board_tensor: torch.Tensor,
        legal_indices: list,
    ) -> torch.Tensor:
        """
        Return a softmax probability distribution over legal moves only.

        Args:
            board_tensor:  (1, NUM_PLANES, 8, 8) — single position, already on device.
            legal_indices: List of int, each in [0, 4095].

        Returns:
            probs: (MOVE_VOCAB,) tensor — non-legal moves have probability 0.
        """
        logits = self.forward(board_tensor).squeeze(0)  # (MOVE_VOCAB,)
        mask = torch.full((MOVE_VOCAB,), float("-inf"), device=logits.device)
        mask[legal_indices] = 0.0
        return F.softmax(logits + mask, dim=0)
