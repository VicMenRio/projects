"""
Funciones de utilidad para visualización y análisis de resultados
"""

import torch
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import List, Optional


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_predictions(model, dataset, num_samples=16, class_names=None, title=None):
    """
    Visualiza predicciones del modelo en una cuadrícula
    
    Args:
        model: Modelo entrenado
        dataset: Dataset de test
        num_samples: Número de muestras a visualizar
        class_names: Lista con nombres de clases
        title: Título para guardar la figura
    """

    if class_names is None:
        class_names = [f'Class {i}' for i in range(10)]

    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()

    for idx, ax in zip(indices, axes):
        image, true_label = dataset[idx]

        with torch.no_grad():
            logits = model(image.unsqueeze(0))
            pred_label = torch.argmax(logits, dim=1).item()

        img = image.squeeze().numpy()
        ax.imshow(img, cmap='gray')

        color = 'green' if pred_label == true_label else 'red'
        ax.set_title(f'True: {class_names[true_label]}\nPred: {class_names[pred_label]}',
                    color=color, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    if title:
        plt.savefig(title, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None, title=None):
    """
    Visualiza matriz de confusión
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones
        class_names: Lista con nombres de clases
        title: Título de la figura
    """

    if class_names is None:
        class_names = [f'Class {i}' for i in range(10)]

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title if title else 'Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    if title:
        plt.savefig(title.replace(' ', '_') + '.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_sample_images(dataset, num_samples=16, class_names=None):
    """
    Visualiza muestras aleatorias del dataset
    
    Args:
        dataset: Dataset KMNIST
        num_samples: Número de muestras a visualizar
        class_names: Lista con nombres de clases
    """
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(10)]
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for idx, ax in zip(indices, axes):
        image, label = dataset[idx]
        img = image.squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.set_title(f'{class_names[label]}', fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_training_history(trainer_history, metric='loss'):
    """
    Visualiza el historial de entrenamiento
    
    Args:
        trainer_history: Historial del trainer de PyTorch Lightning
        metric: Métrica a visualizar ('loss' o 'acc')
    """
    
    plt.figure(figsize=(10, 6))
    
    if metric == 'loss':
        plt.plot(trainer_history['train_loss'], label='Train Loss')
        plt.plot(trainer_history['val_loss'], label='Val Loss')
        plt.ylabel('Loss')
    else:
        plt.plot(trainer_history['train_acc'], label='Train Accuracy')
        plt.plot(trainer_history['val_acc'], label='Val Accuracy')
        plt.ylabel('Accuracy')
    
    plt.xlabel('Epoch')
    plt.title(f'Training History - {metric.capitalize()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# DATA UTILITIES
# ============================================================================

def get_kmnist_dataset(train=False, transform=None):
    """
    Carga el dataset KMNIST
    
    Args:
        train: Si True, carga el set de entrenamiento
        transform: Transformaciones a aplicar
    
    Returns:
        Dataset KMNIST
    """
    
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    return torchvision.datasets.KMNIST(
        root='./data',
        train=train,
        download=True,
        transform=transform
    )


def get_class_distribution(dataset):
    """
    Calcula la distribución de clases en el dataset
    
    Args:
        dataset: Dataset KMNIST
    
    Returns:
        Dict con la distribución de clases
    """
    
    labels = [label for _, label in dataset]
    unique, counts = np.unique(labels, return_counts=True)
    
    return dict(zip(unique.tolist(), counts.tolist()))


def plot_class_distribution(dataset, class_names=None):
    """
    Visualiza la distribución de clases
    
    Args:
        dataset: Dataset KMNIST
        class_names: Lista con nombres de clases
    """
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(10)]
    
    distribution = get_class_distribution(dataset)
    
    plt.figure(figsize=(10, 6))
    plt.bar([class_names[i] for i in distribution.keys()], distribution.values())
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ============================================================================
# MODEL UTILITIES
# ============================================================================

def count_parameters(model):
    """
    Cuenta el número de parámetros entrenables del modelo
    
    Args:
        model: Modelo PyTorch
    
    Returns:
        Número total de parámetros
    """
    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model):
    """
    Obtiene un resumen del modelo
    
    Args:
        model: Modelo PyTorch
    
    Returns:
        Dict con información del modelo
    """
    
    total_params = count_parameters(model)
    
    summary = {
        'Total Parameters': total_params,
        'Model Type': model.__class__.__name__,
        'Trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'Non-Trainable': sum(p.numel() for p in model.parameters() if not p.requires_grad)
    }
    
    return summary


def print_model_summary(model):
    """
    Imprime un resumen del modelo
    
    Args:
        model: Modelo PyTorch
    """
    
    summary = get_model_summary(model)
    
    print("\n" + "="*50)
    print("MODEL SUMMARY")
    print("="*50)
    for key, value in summary.items():
        print(f"{key}: {value:,}")
    print("="*50 + "\n")


# ============================================================================
# KMNIST CONSTANTS
# ============================================================================

# Nombres de las clases KMNIST (caracteres Hiragana)
KMNIST_CLASSES = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo']


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    # Ejemplo de uso de las funciones
    
    # Cargar dataset
    test_dataset = get_kmnist_dataset(train=False)
    
    # Visualizar muestras
    print("Visualizando muestras del dataset...")
    visualize_sample_images(test_dataset, num_samples=16, class_names=KMNIST_CLASSES)
    
    # Mostrar distribución de clases
    print("\nDistribución de clases:")
    plot_class_distribution(test_dataset, class_names=KMNIST_CLASSES)
