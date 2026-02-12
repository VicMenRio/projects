"""
Modelos de Deep Learning para clasificación KMNIST
Incluye: Baseline MLP, SimpleCNN, ResNetCNN, DenseNetCNN, InceptionCNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import List


# ============================================================================
# BASELINE MODEL - MLP
# ============================================================================

class BaselineMLP(pl.LightningModule):
    """Baseline MLP según especificaciones del proyecto"""

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc2 = nn.Linear(100, 10)

        self.criterion = nn.CrossEntropyLoss()
        self.test_predictions = []
        self.test_targets = []

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        self.test_predictions.extend(preds.cpu().numpy())
        self.test_targets.extend(y.cpu().numpy())
        return preds

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


# ============================================================================
# BUILDING BLOCKS
# ============================================================================

class ConvBlock(nn.Module):
    """Bloque convolucional básico configurable"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        return x


class ResidualBlock(nn.Module):
    """Bloque con Skip Connection configurable"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.0):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, kernel_size,
                                   padding, dropout)

        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return F.relu(self.conv_block(x) + self.skip(x))


class ResidualBlockIntermediate(nn.Module):
    """Skip Connection con capas intermedias configurables"""

    def __init__(self, in_channels, out_channels, intermediate_channels,
                 kernel_size=3, padding=1, dropout=0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.dropout1 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv2d(intermediate_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)

        out += self.skip(identity)
        out = F.relu(out)
        return out


class InceptionBlock(nn.Module):
    """Bloque Inception configurable"""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        branch_channels = out_channels // 4

        # 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )

        # 1x1 -> 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(),
            nn.Conv2d(branch_channels, branch_channels, 3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )

        # 1x1 -> 5x5 convolution
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(),
            nn.Conv2d(branch_channels, branch_channels, 5, padding=2),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )

        # MaxPool -> 1x1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, 1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)


# ============================================================================
# CNN ARCHITECTURES
# ============================================================================

class SimpleCNN(pl.LightningModule):
    """CNN simple con bloques convolucionales configurables"""

    def __init__(self, channels: List[int] = [64, 128, 256],
                 kernel_size: int = 3,
                 pool_type: str = 'max',
                 pool_size: int = 2,
                 dense_layers: List[int] = [],
                 dropout_conv: float = 0.0,
                 dropout_dense: float = 0.5,
                 lr: float = 0.001,
                 optimizer: str = 'adam',
                 weight_decay: float = 0.0):
        super().__init__()
        self.save_hyperparameters()

        layers = []
        in_channels = 1
        padding = kernel_size // 2

        for out_channels in channels:
            layers.append(ConvBlock(in_channels, out_channels, kernel_size,
                                   padding, dropout_conv))

            if pool_type == 'max':
                layers.append(nn.MaxPool2d(pool_size))
            elif pool_type == 'avg':
                layers.append(nn.AvgPool2d(pool_size))

            in_channels = out_channels

        layers.append(nn.AdaptiveAvgPool2d(1))
        self.cnn = nn.Sequential(*layers)

        classifier_layers = [nn.Flatten()]
        prev_size = channels[-1]
        for dense_size in dense_layers:
            classifier_layers.append(nn.Linear(prev_size, dense_size))
            classifier_layers.append(nn.ReLU())
            if dropout_dense > 0:
                classifier_layers.append(nn.Dropout(dropout_dense))
            prev_size = dense_size

        classifier_layers.append(nn.Linear(prev_size, 10))
        self.classifier = nn.Sequential(*classifier_layers)

        self.criterion = nn.CrossEntropyLoss()
        self.test_predictions = []
        self.test_targets = []

    def forward(self, x):
        x = self.cnn(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        self.test_predictions.extend(preds.cpu().numpy())
        self.test_targets.extend(y.cpu().numpy())
        return preds

    def configure_optimizers(self):
        if self.hparams.optimizer == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr,
                                   weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'adamw':
            return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                    weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.hparams.lr,
                                  momentum=0.9, weight_decay=self.hparams.weight_decay)


class ResNetCNN(pl.LightningModule):
    """CNN con Skip Connections configurables"""

    def __init__(self, channels: List[int] = [32, 64, 128],
                 kernel_size: int = 3,
                 pool_type: str = 'max',
                 pool_size: int = 2,
                 dense_layers: List[int] = [],
                 dropout_conv: float = 0.0,
                 dropout_dense: float = 0.5,
                 lr: float = 0.001,
                 optimizer: str = 'adam',
                 weight_decay: float = 0.0):
        super().__init__()
        self.save_hyperparameters()

        padding = kernel_size // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size, padding=padding),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU()
        )

        layers = []
        for i in range(len(channels)):
            in_ch = channels[i]
            out_ch = channels[i]

            layers.append(ResidualBlock(in_ch, out_ch, kernel_size, padding, dropout_conv))

            if i < len(channels) - 1:
                next_ch = channels[i + 1]
                layers.append(ResidualBlock(out_ch, next_ch, kernel_size, padding, dropout_conv))

                if pool_type == 'max':
                    layers.append(nn.MaxPool2d(pool_size))
                elif pool_type == 'avg':
                    layers.append(nn.AvgPool2d(pool_size))

        layers.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*layers)

        classifier_layers = [nn.Flatten()]
        prev_size = channels[-1]

        for dense_size in dense_layers:
            classifier_layers.append(nn.Linear(prev_size, dense_size))
            classifier_layers.append(nn.ReLU())
            if dropout_dense > 0:
                classifier_layers.append(nn.Dropout(dropout_dense))
            prev_size = dense_size

        classifier_layers.append(nn.Linear(prev_size, 10))
        self.classifier = nn.Sequential(*classifier_layers)

        self.criterion = nn.CrossEntropyLoss()
        self.test_predictions = []
        self.test_targets = []

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        self.test_predictions.extend(preds.cpu().numpy())
        self.test_targets.extend(y.cpu().numpy())
        return preds

    def configure_optimizers(self):
        if self.hparams.optimizer == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr,
                                   weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'adamw':
            return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                    weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.hparams.lr,
                                  momentum=0.9, weight_decay=self.hparams.weight_decay)


class DenseNetCNN(pl.LightningModule):
    """CNN con Skip Connections intermedias configurables"""

    def __init__(self, channels: List[int] = [64, 128, 256],
                 intermediate_ratio: float = 0.75,
                 kernel_size: int = 3,
                 pool_type: str = 'max',
                 pool_size: int = 2,
                 dense_layers: List[int] = [],
                 dropout_conv: float = 0.0,
                 dropout_dense: float = 0.5,
                 lr: float = 0.001,
                 optimizer: str = 'adam',
                 weight_decay: float = 0.0):
        super().__init__()
        self.save_hyperparameters()

        padding = kernel_size // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        layers = []
        in_ch = 32

        for out_ch in channels:
            intermediate_ch = int(out_ch * intermediate_ratio)
            layers.append(ResidualBlockIntermediate(in_ch, out_ch, intermediate_ch,
                                                   kernel_size, padding, dropout_conv))

            if pool_type == 'max':
                layers.append(nn.MaxPool2d(pool_size))
            elif pool_type == 'avg':
                layers.append(nn.AvgPool2d(pool_size))

            in_ch = out_ch

        layers.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*layers)

        classifier_layers = [nn.Flatten()]
        prev_size = channels[-1]

        for dense_size in dense_layers:
            classifier_layers.append(nn.Linear(prev_size, dense_size))
            classifier_layers.append(nn.ReLU())
            if dropout_dense > 0:
                classifier_layers.append(nn.Dropout(dropout_dense))
            prev_size = dense_size

        classifier_layers.append(nn.Linear(prev_size, 10))
        self.classifier = nn.Sequential(*classifier_layers)

        self.criterion = nn.CrossEntropyLoss()
        self.test_predictions = []
        self.test_targets = []

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        self.test_predictions.extend(preds.cpu().numpy())
        self.test_targets.extend(y.cpu().numpy())
        return preds

    def configure_optimizers(self):
        if self.hparams.optimizer == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr,
                                   weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'adamw':
            return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                    weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.hparams.lr,
                                  momentum=0.9, weight_decay=self.hparams.weight_decay)


class InceptionCNN(pl.LightningModule):
    """CNN con bloques Inception configurables"""

    def __init__(self, channels: List[int] = [64, 128, 256],
                 kernel_size=3,
                 pool_type: str = 'max',
                 dense_layers: List[int] = [],
                 dropout_conv: float = 0.0,
                 dropout_dense: float = 0.5,
                 lr: float = 0.001,
                 optimizer: str = 'adam',
                 weight_decay: float = 0.0):
        super().__init__()
        self.save_hyperparameters()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        layers = []
        in_ch = 32

        for out_ch in channels:
            layers.append(InceptionBlock(in_ch, out_ch, dropout_conv))

            if pool_type == 'max':
                layers.append(nn.MaxPool2d(2))
            elif pool_type == 'avg':
                layers.append(nn.AvgPool2d(2))

            in_ch = out_ch

        layers.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*layers)

        classifier_layers = [nn.Flatten()]
        prev_size = channels[-1]

        for dense_size in dense_layers:
            classifier_layers.append(nn.Linear(prev_size, dense_size))
            classifier_layers.append(nn.ReLU())
            if dropout_dense > 0:
                classifier_layers.append(nn.Dropout(dropout_dense))
            prev_size = dense_size

        classifier_layers.append(nn.Linear(prev_size, 10))
        self.classifier = nn.Sequential(*classifier_layers)

        self.criterion = nn.CrossEntropyLoss()
        self.test_predictions = []
        self.test_targets = []

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        self.test_predictions.extend(preds.cpu().numpy())
        self.test_targets.extend(y.cpu().numpy())
        return preds

    def configure_optimizers(self):
        if self.hparams.optimizer == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr,
                                   weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'adamw':
            return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                    weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.hparams.lr,
                                  momentum=0.9, weight_decay=self.hparams.weight_decay)
