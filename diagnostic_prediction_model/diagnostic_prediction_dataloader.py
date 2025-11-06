"""Dataset utilities for error prediction inputs."""

from torch.utils.data import Dataset, DataLoader

class DiagnosticPredictionDataset(Dataset):
    """Dataset wrapper for error prediction data.

    Expects ``data_dir`` to be an indexable sequence (e.g., list) of items.
    """

    def __init__(self, data_dir: str):
        """Initialize the dataset.

        Args:
            data_dir: Container providing indexable items.
        """
        self.data_dir = data_dir

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.data_dir)

    def __getitem__(self, idx):
        """Retrieve an item by index.

        Args:
            idx: Integer index of the item.

        Returns:
            The item at the specified index.
        """
        return self.data_dir[idx]

    @property
    def classes(self):
        """Return the classes in the dataset."""
        return self.data_dir.classes()


