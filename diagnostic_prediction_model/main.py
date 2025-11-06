"""Simple PyTorch training framework for diagnostic prediction."""

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from diagnostic_prediction_dataloader import DiagnosticPredictionDataset


def load_data(json_path):
    """Load diagnostic data from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def main():
    data = load_data("data/diagnostic_prediction.json")

if __name__ == "__main__":
    main()

