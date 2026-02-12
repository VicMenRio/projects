"""
KMNIST Classification Project
Deep Learning para clasificación de caracteres japoneses antiguos
"""

from .data_module import KMNISTDataModule
from .models import (
    BaselineMLP,
    SimpleCNN,
    ResNetCNN,
    DenseNetCNN,
    InceptionCNN,
    ConvBlock,
    ResidualBlock,
    ResidualBlockIntermediate,
    InceptionBlock
)
from .utils import (
    visualize_predictions,
    plot_confusion_matrix,
    visualize_sample_images,
    plot_training_history,
    get_kmnist_dataset,
    get_class_distribution,
    plot_class_distribution,
    count_parameters,
    get_model_summary,
    print_model_summary,
    KMNIST_CLASSES
)

__version__ = '1.0.0'
__author__ = 'Victor Méndez'

__all__ = [
    # Data
    'KMNISTDataModule',
    
    # Models
    'BaselineMLP',
    'SimpleCNN',
    'ResNetCNN',
    'DenseNetCNN',
    'InceptionCNN',
    
    # Building blocks
    'ConvBlock',
    'ResidualBlock',
    'ResidualBlockIntermediate',
    'InceptionBlock',
    
    # Utilities
    'visualize_predictions',
    'plot_confusion_matrix',
    'visualize_sample_images',
    'plot_training_history',
    'get_kmnist_dataset',
    'get_class_distribution',
    'plot_class_distribution',
    'count_parameters',
    'get_model_summary',
    'print_model_summary',
    'KMNIST_CLASSES',
]
