# KMNIST Classification Project

Proyecto de Deep Learning para clasificación de caracteres KMNIST (Kuzushiji-MNIST) utilizando PyTorch Lightning.

## Descripción

Este proyecto implementa y compara diferentes arquitecturas de redes neuronales para la clasificación de caracteres japoneses antiguos (Kuzushiji). Se incluyen:

- **Baseline MLP**: Perceptrón multicapa simple
- **SimpleCNN**: Red convolucional básica
- **ResNetCNN**: CNN con conexiones residuales
- **DenseNetCNN**: CNN con bloques densos
- **InceptionCNN**: CNN con bloques Inception

## Estructura del Proyecto

```
├── requirements.txt          # Dependencias del proyecto
├── src/
│   ├── data_module.py       # DataModule para KMNIST
│   ├── models.py            # Arquitecturas de redes neuronales
│   ├── train.py             # Scripts de entrenamiento y optimización
│   └── utils.py             # Utilidades y visualización
└── README.md                # Este archivo
```

## Instalación

1. Clonar el repositorio:
```bash
git clone <tu-repositorio>
cd <nombre-repositorio>
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### Entrenamiento Completo

Para ejecutar el pipeline completo de optimización y entrenamiento:

```bash
python src/train.py
```

Este comando:
1. Optimiza hiperparámetros del Baseline MLP usando Optuna
2. Optimiza hiperparámetros de las CNN usando Optuna
3. Entrena los modelos con los mejores parámetros
4. Evalúa en validation y test sets
5. Genera visualizaciones

## Dataset

El proyecto utiliza el dataset **KMNIST** (Kuzushiji-MNIST):
- 70,000 imágenes de 28x28 píxeles en escala de grises
- 10 clases de caracteres Hiragana
- 60,000 imágenes de entrenamiento
- 10,000 imágenes de test
- Split adicional 80/20 para validación

### Clases KMNIST

| Índice | Carácter | Romanización |
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

## Arquitecturas

### Baseline MLP
- Capa de entrada: 784 (28×28 aplanado)
- Capa oculta: 100 neuronas + ReLU
- Capa de salida: 10 clases
- Optimizador: SGD

### SimpleCNN
- Bloques convolucionales configurables
- Batch Normalization
- Pooling (Max/Average)
- Dropout configurable
- Clasificador denso

### ResNetCNN
- Conexiones residuales (skip connections)
- Bloques residuales con BN
- Facilita entrenamiento profundo
- Mitiga vanishing gradient

### DenseNetCNN
- Conexiones residuales intermedias
- Bloques con canales intermedios configurables
- Mayor expresividad

### InceptionCNN
- Bloques Inception multiescala
- Convoluciones paralelas (1×1, 3×3, 5×5)
- Captura características a múltiples escalas

## Optimización de Hiperparámetros

- Arquitectura: [SimpleCNN, ResNetCNN, DenseNetCNN, InceptionCNN]
- Número de capas: 1-3
- Kernel size: [3, 5]
- Pool type: [max, avg]
- Dropout conv: [0.0, 0.15, 0.3]
- Dropout dense: [0.0, 0.25, 0.5]
- Learning rate: [0.01, 0.001, 0.0001]
- Optimizer: [adam, adamw, sgd]
- Weight decay: [0.001, 0.0001, 0]
- Método: TPE (Tree-structured Parzen Estimator)

## Data Augmentation

Para las CNN se aplica:
- Rotación aleatoria (±15°)
- Flip horizontal (p=0.2)
- Flip vertical (p=0.2)
- Inversión de colores (p=0.2)
- Normalización estándar

## Callbacks

- **Early Stopping**: Paciencia de 5 épocas
- **Model Checkpoint**: Guarda mejor modelo según val_loss

## Métricas

- **Loss**: CrossEntropyLoss
- **Accuracy**: Precisión general
- **F1-Score**: F1 ponderado por clase
- **Confusion Matrix**: Matriz de confusión

## Visualizaciones

El proyecto genera:
- Matrices de confusión
- Predicciones visuales con ejemplos correctos/incorrectos
- Historial de optimización Optuna
- Importancia de hiperparámetros
- Gráficos de coordenadas paralelas

## Requisitos

- Python 3.8+
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- Optuna 3.3+
- NumPy, Matplotlib, Seaborn, scikit-learn

## Autor

Victor Méndez

## Licencia

Este proyecto es de código abierto y está disponible para uso académico y educativo.
