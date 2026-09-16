"""1D Convolutional Neural Network (1D CNN) Skeleton for Signal Data.

Educational Simulation System.
Applies 1D Conv, BatchNorm, MaxPool, Dropout, and Linear layers for ECG-like signal pattern classification.
"""

from typing import Dict, Any

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    ModuleBase = nn.Module
except ImportError:
    TORCH_AVAILABLE = False
    ModuleBase = object

from src.utils.logger import setup_logger

logger = setup_logger("cnn_1d_model")


class Physio1DCNN(ModuleBase):
    """1D CNN architecture for physiological signal pattern classification."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4, kernel_size: int = 5):
        """Initialize 1D CNN architecture.

        Args:
            in_channels: Input channels (default: 1 for single ECG channel).
            num_classes: Output class logits count (default: 4 pattern categories).
            kernel_size: Convolution kernel filter size.
        """
        if TORCH_AVAILABLE:
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=kernel_size, padding=kernel_size // 2)
            self.bn1 = nn.BatchNorm1d(16)
            self.relu = nn.ReLU()
            self.pool1 = nn.MaxPool1d(2)

            self.conv2 = nn.Conv1d(16, 32, kernel_size=kernel_size, padding=kernel_size // 2)
            self.bn2 = nn.BatchNorm1d(32)
            self.pool2 = nn.MaxPool1d(2)

            self.adaptive_pool = nn.AdaptiveAvgPool1d(16)
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(32 * 16, num_classes)
        else:
            logger.warning("PyTorch is not installed in current environment. Physio1DCNN initialized in fallback mode.")

        logger.info(f"Initialized Physio1DCNN (in_channels={in_channels}, num_classes={num_classes})")

    def forward(self, x: Any) -> Any:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, in_channels, signal_length).

        Returns:
            Output logit tensor of shape (batch_size, num_classes).
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for forward pass.")

        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        out = self.fc(x)
        return out
