"""
Script de entrenamiento para modelos KMNIST
Incluye optimización con Optuna y entrenamiento final
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from data_module import KMNISTDataModule
from models import BaselineMLP, SimpleCNN, ResNetCNN, DenseNetCNN, InceptionCNN


# ============================================================================
# BASELINE TRAINING AND EVALUATION
# ============================================================================
def train_eval_mlp():

   # Crear modelo
    model = BaselineMLP()

    # Data
    dm = KMNISTDataModule(batch_size=8, augment=False)

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')

    # Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[early_stop],
        accelerator='auto',
        devices=1,
        enable_progress_bar=True,
        logger=True
    )

    # Entrenar
    trainer.fit(model, dm)

    # Cargar mejor modelo
    best_model = model_class.load_from_checkpoint(checkpoint.best_model_path)

    # Evaluar en validation
    print(f"\nEvaluación para VALIDATION set...")
    trainer.validate(best_model, dm)

    # Para validation necesitamos hacer predicciones manualmente
    best_model.eval()
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in dm.val_dataloader():
            x, y = batch
            logits = best_model(x)
            preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(y.cpu().numpy())

    y_true = np.array(val_targets)
    y_pred = np.array(val_preds)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    val_results = {
        'Model': 'BaselineMLP',
        'Eval Set': 'Validation',
        'Accuracy': accuracy,
        'F1-Score': f1,
        'Val Loss': checkpoint.best_model_score.item(),
        'y_true': y_true,
        'y_pred': y_pred,
        **config
    }

    # Evaluar en test
    print(f"\nEvaluación para TEST set...")
    trainer.test(best_model, dm)

    # Calcular métricas de test
    y_true = np.array(best_model.test_targets)
    y_pred = np.array(best_model.test_predictions)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    test_results = {
        'Model': 'BaselineMLP',
        'Eval Set': 'Test',
        'Accuracy': accuracy,
        'F1-Score': f1,
        'Val Loss': checkpoint.best_model_score.item(),
        'y_true': y_true,
        'y_pred': y_pred,
        **config
    }

    return val_results, test_results, best_model

# ============================================================================
# OPTUNA OPTIMIZATION
# ============================================================================

def run_optuna_study(study_name, n_trials=10):
    """Ejecuta un estudio de Optuna para un modelo"""

    print(f"\n{'='*80}")
    print(f"OPTIMIZACIÓN CON OPTUNA: {study_name}")
    print(f"Número de trials: {n_trials}")
    print(f"{'='*80}\n")

    # Crear estudio
    sampler =  optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner()
    )

    # Optimizar
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Mostrar resultados
    print(f"\n{'='*80}")
    print(f"OPTIMIZACIÓN COMPLETADA: {study_name}")
    print(f"{'='*80}")
    print(f"Mejor trial: {study.best_trial.number}")
    print(f"Mejor valor (val_loss): {study.best_value:.4f}")
    print(f"\nMejores parámetros:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    return study


def generate_optuna_visualizations(study, study_name):
    """Genera visualizaciones de Optuna"""

    print(f"\nGenerando visualizaciones de Optuna para {study_name}...")

    # 1. Optimization History
    fig = plot_optimization_history(study)
    fig.show()

    # 2. Parameter Importances
    fig = plot_param_importances(study)
    fig.show()

    # 3. Parallel Coordinate Plot
    fig = plot_parallel_coordinate(study)
    fig.show()

    # 4. Slice Plot
    fig = plot_slice(study)
    fig.show()


# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

arquitecture_dict = {
    'SimpleCNN': SimpleCNN,
    'ResNetCNN': ResNetCNN,
    'DenseNetCNN': DenseNetCNN,
    'InceptionCNN': InceptionCNN
}


def objective(trial):
    """Función objetivo con Optuna"""

    # Sugerir hiperparámetros
    n_layers = trial.suggest_int('n_layers', 1, 3)
    channels = [64 * (2**i) for i in range(n_layers)]

    kernel_size = trial.suggest_categorical('kernel_size', [3, 5])
    pool_type = trial.suggest_categorical('pool_type', ['max', 'avg'])

    n_dense = trial.suggest_int('n_dense', 2, 4)
    dense_layers = [512 // (2**i) for i in range(n_dense)]

    dropout_conv = trial.suggest_categorical('dropout_conv', [0.0, 0.15, 0.3])
    dropout_dense = trial.suggest_categorical('dropout_dense', [0.0, 0.25, 0.5])

    lr = trial.suggest_categorical('lr', [0.01, 0.001, 0.0001])
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])
    weight_decay = trial.suggest_categorical('weight_decay', [0.001, 0.0001, 0])

    # Crear modelo
    arquitecture = trial.suggest_categorical('architecture', ['SimpleCNN', 'ResNetCNN', 'DenseNetCNN', 'InceptionCNN'])
    ModelClass = arquitecture_dict[arquitecture]

    model = ModelClass(
        channels=channels,
        kernel_size=kernel_size,
        pool_type=pool_type,
        dense_layers=dense_layers,
        dropout_conv=dropout_conv,
        dropout_dense=dropout_dense,
        lr=lr,
        optimizer=optimizer,
        weight_decay=weight_decay
    )

    # Data
    dm = KMNISTDataModule(batch_size=32, augment=True)

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')

    # Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[early_stop],
        accelerator='auto',
        devices=1,
        enable_progress_bar=True,
        logger=True
    )

    # Entrenar
    trainer.fit(model, dm)

    # Obtener mejor val_loss
    return trainer.callback_metrics['val_loss'].item()


# ============================================================================
# FINAL TRAINING WITH BEST PARAMETERS
# ============================================================================

def train_model_with_best_params(study_name, best_params,
                                  batch_size=32, augment=True, max_epochs=15):
    """Entrena modelo con mejores parámetros encontrados por Optuna"""

    print(f"\n{'='*80}")
    print(f"ENTRENAMIENTO FINAL: {study_name}")
    print(f"{'='*80}")
    print(f"Configuración:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Preparar parámetros
    config = best_params.copy()
    model_name = config.pop('architecture')

    n_layers = config.pop('n_layers')
    config['channels'] = [64 * (2**i) for i in range(n_layers)]

    n_dense = config.pop('n_dense')
    config['dense_layers'] = [512 // (2**i) for i in range(n_dense)]

    # Data
    dm = KMNISTDataModule(batch_size=batch_size, augment=augment)

    # Modelo
    model_class = arquitecture_dict[model_name]
    model = model_class(**config)

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        filename=f'{model_name}_best'
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[early_stop, checkpoint],
        accelerator='auto',
        devices=1,
        log_every_n_steps=10,
        enable_progress_bar=True,
        logger=True
    )

    # Entrenar
    trainer.fit(model, dm)

    # Cargar mejor modelo
    best_model = model_class.load_from_checkpoint(checkpoint.best_model_path)

    # Evaluar en validation
    print(f"\nEvaluación para VALIDATION set...")
    trainer.validate(best_model, dm)

    # Para validation necesitamos hacer predicciones manualmente
    best_model.eval()
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in dm.val_dataloader():
            x, y = batch
            logits = best_model(x)
            preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(y.cpu().numpy())

    y_true = np.array(val_targets)
    y_pred = np.array(val_preds)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    val_results = {
        'Model': model_name,
        'Eval Set': 'Validation',
        'Accuracy': accuracy,
        'F1-Score': f1,
        'Val Loss': checkpoint.best_model_score.item(),
        'y_true': y_true,
        'y_pred': y_pred,
        **config
    }

    # Evaluar en test
    print(f"\nEvaluación para TEST set...")
    trainer.test(best_model, dm)

    # Calcular métricas de test
    y_true = np.array(best_model.test_targets)
    y_pred = np.array(best_model.test_predictions)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    test_results = {
        'Model': model_name,
        'Eval Set': 'Test',
        'Accuracy': accuracy,
        'F1-Score': f1,
        'Val Loss': checkpoint.best_model_score.item(),
        'y_true': y_true,
        'y_pred': y_pred,
        **config
    }

    return val_results, test_results, best_model


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

if __name__ == '__main__':
    # Entrenar Baseline MLP
    print("\n" + "="*80)
    print("FASE 1: BASELINE MLP")
    print("="*80)

    mlp_val_results, mlp_test_results, mlp_best_model = train_eval_mlp()

    # Entrenar CNN
    print("\n" + "="*80)
    print("FASE 2: OPTIMIZACIÓN CNN")
    print("="*80)
    
    cnn_study = run_optuna_study('CNN_Optimization', n_trials=10)
    generate_optuna_visualizations(cnn_study, 'CNN_Optimization')
    
    cnn_val_results, cnn_test_results, cnn_best_model = train_model_with_best_params(
        'CNN_Optimization',
        cnn_study.best_params,
        batch_size=32,
        augment=True,
        max_epochs=10,
    )

    # Mostrar resultados finales
    print("\n" + "="*80)
    print("RESULTADOS FINALES")
    print("="*80)
    
    import pandas as pd
    all_results = [mlp_val_results, mlp_test_results, cnn_val_results, cnn_test_results]
    df_results = pd.DataFrame(all_results)
    
    key_cols = ['Model', 'Eval Set', 'Accuracy', 'F1-Score', 'Val Loss']
    print("\n" + df_results[key_cols].to_string(index=False))
