import lightning as L
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from plcls.augmentations import get_transforms


class TorchVisionDataModule(L.LightningDataModule):
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset

    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.data_dir = conf.data.data_dir

        self.train_transform = get_transforms(
            augmentations=conf.augmentations.type,
            augs_prob=conf.augmentations.probability,
            return_lambda=True,
        )
        self.val_transform = get_transforms(return_lambda=True)

        self.data_class = getattr(datasets, conf.data.data_class)

    def prepare_data(self):
        """Download datasets."""
        self.data_class(self.data_dir, train=True, download=True)
        self.data_class(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        """Assign train/val/test datasets for use in dataloaders."""
        if stage == "fit":
            self.train_dataset = self.data_class(
                self.data_dir, train=True, transform=self.train_transform
            )

            print(f"Train dataset: {len(self.train_dataset)}")

            self.val_dataset = self.data_class(
                self.data_dir, train=False, transform=self.val_transform
            )

            print(f"Valid dataset: {len(self.val_dataset)}")

        if stage == "test":
            self.test_dataset = self.data_class(
                self.data_dir, train=False, transform=self.val_transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.conf.data.train_loader_params)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.conf.data.valid_loader_params)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.conf.data.test_loader_params)
