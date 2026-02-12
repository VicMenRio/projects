"""
Data Module para el dataset KMNIST
Maneja la carga, transformación y preparación de datos
"""

import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
import pytorch_lightning as pl


class KMNISTDataModule(pl.LightningDataModule):
    """DataModule para KMNIST con data augmentation avanzado"""

    def __init__(self, batch_size: int = 8, augment: bool = False, num_workers: int = 2):
        super().__init__()
        self.batch_size = batch_size
        self.augment = augment
        self.num_workers = num_workers

        # Transforms básicos (baseline y validación/test)
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Transforms con augmentation (para modelos avanzados)
        self.augment_transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomVerticalFlip(p=0.2),
            torchvision.transforms.RandomInvert(p=0.2),  # Invertir colores
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def prepare_data(self):
        """Descarga los datos si no existen"""
        torchvision.datasets.KMNIST(root='./data', train=True, download=True)
        torchvision.datasets.KMNIST(root='./data', train=False, download=True)

    def setup(self, stage=None):
        """Configura los datasets de entrenamiento, validación y test"""
        full_train = torchvision.datasets.KMNIST(
            root='./data',
            train=True,
            transform=None,
        )

        # Split train/validation (80/20)
        train_size = int(0.8 * len(full_train))
        val_size = len(full_train) - train_size
        self.train_dataset, self.val_dataset = random_split(
            full_train, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Aplicar transformaciones
        if self.augment:
            self.train_dataset.dataset.transform = self.augment_transform
        else:
            self.train_dataset.dataset.transform = self.basic_transform
        self.val_dataset.dataset.transform = self.basic_transform

        # Dataset de test
        self.test_dataset = torchvision.datasets.KMNIST(
            root='./data',
            train=False,
            transform=self.basic_transform
        )

    def train_dataloader(self):
        """DataLoader para entrenamiento"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=2
        )

    def val_dataloader(self):
        """DataLoader para validación"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=2
        )

    def test_dataloader(self):
        """DataLoader para test"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=2
        )
