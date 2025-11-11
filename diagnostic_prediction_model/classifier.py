# DiagnosticClassifier.py
# Description: A classifier for diagnostic prediction.
# Author: Robert Doherty
# Date: 2025-11-08

import torch
import torch.nn as nn


class DiagnosticClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DiagnosticClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)