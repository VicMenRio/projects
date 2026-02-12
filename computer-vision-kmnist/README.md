# KMNIST Classification Project

Deep Learning project for KMNIST (Kuzushiji-MNIST) character classification using PyTorch Lightning.

## Description

This project implements and compares different neural network architectures for the classification of ancient Japanese characters (Kuzushiji). The following models are included:

- **Baseline MLP**: Simple multilayer perceptron
- **SimpleCNN**: Basic convolutional neural network
- **ResNetCNN**: CNN with residual connections
- **DenseNetCNN**: CNN with dense-style blocks
- **InceptionCNN**: CNN with Inception modules

## Project Structure
├── requirements.txt # Project dependencies

├── src/

│ ├── data_module.py # KMNIST DataModule

│ ├── models.py # Neural network architectures

│ ├── train.py # Training and optimization scripts

│ ├── utils.py # Utilities and visualization

| └── predict.py # Simple inference script

└── README.md # This file

## Installation

1. Clone the repository:
```bash
git clone <VicMenRio/projects/computer-vision-kmnist>
cd <VicMenRio/projects/computer-vision-kmnis>

2. Install dependencies:
pip install -r requirements.txt

## Usage
### Full Training Pipeline
To run the complete optimization and training pipeline:
python src/train.py

This command will:
1. Optimize Baseline MLP hyperparameters using Optuna
2. Optimize CNN hyperparameters using Optuna
3. Train models with the best parameters
4. Evaluate on validation and test sets
5. Generate visualizations

Dataset
This project uses the KMNIST (Kuzushiji-MNIST) dataset:
- 70,000 grayscale images of size 28x28 pixels
- 10 Hiragana character classes
- 60,000 training images
- 10,000 test images
- Additional 80/20 validation split

### Clases KMNIST
| Index  | Character| Romazatation |
|--------|----------|--------------|
| 0      | お       | o            |
| 1      | き       | ki           |
| 2      | す       | su           |
| 3      | つ       | tsu          |
| 4      | な       | na           |
| 5      | は       | ha           |
| 6      | ま       | ma           |
| 7      | や       | ya           |
| 8      | れ       | re           |
| 9      | を       | wo           |

## Architectures
### Baseline MLP
- Input layer: 784 (flattened 28×28)
- Hidden layer: 100 neurons + ReLU
- Output layer: 10 classes
- Optimizer: SGD

### SimpleCNN
- Configurable convolutional blocks
- Batch Normalization
- Pooling (Max/Average)
- Configurable dropout
- Dense classifier

### ResNetCNN
- Residual connections (skip connections)
- Residual blocks with batch normalization
- Facilitates deeper training
- Mitigates vanishing gradient

### DenseNetCNN
- Intermediate residual connections
- Configurable intermediate channels
- Increased model expressiveness

### InceptionCNN
- Multi-scale Inception blocks
- Parallel convolutions (1×1, 3×3, 5×5)
- Multi-scale feature extraction

## Hyperparameter Optimization
- Architecture: [SimpleCNN, ResNetCNN, DenseNetCNN, InceptionCNN]
- Number of layers: 1–3
- Kernel size: [3, 5]
- Pool type: [max, avg]
- Dropout conv: [0.0, 0.15, 0.3]
- Dropout dense: [0.0, 0.25, 0.5]
- Learning rate: [0.01, 0.001, 0.0001]
- Optimizer: [adam, adamw, sgd]
- Weight decay: [0.001, 0.0001, 0]
- Method: TPE (Tree-structured Parzen Estimator)

### Data Augmentation
- Random rotation (±15°)
- Color inversion (p=0.2)
- Standard normalization

### Callbacks
- Early Stopping: patience of 5 epochs
- Model Checkpoint: saves best model based on val_loss

### Metrics
- Loss: CrossEntropyLoss
- Accuracy: overall accuracy
- F1-Score: weighted F1-score
- Confusion Matrix

### Visualizations
- Confusion matrices
- Visual prediction examples (correct/incorrect samples)
- Optuna optimization history
- Hyperparameter importance plots
- Parallel coordinate plots

### Requirements
- Python 3.8+
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- Optuna 3.3+
- NumPy, Matplotlib, Seaborn, scikit-learn

### Author
Victor Méndez

### License
This project is open-source and available for academic and educational use.
