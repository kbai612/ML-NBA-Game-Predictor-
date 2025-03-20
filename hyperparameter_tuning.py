import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime

def objective_xgboost(trial, X_train, y_train, X_val, y_val, metric='accuracy'):
    """
    Objective function for Optuna optimization of XGBoost hyperparameters
    
    Parameters:
    - trial: Optuna trial object
    - X_train: Training features
    - y_train: Training target
    - X_val: Validation features
    - y_val: Validation target
    - metric: Metric to optimize ('accuracy', 'roc_auc', 'f1', 'log_loss')
    
    Returns:
    - score: Optimization metric score
    """
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        
        # Common parameters
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        
        # Tree-specific parameters
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        
        # Dart specific parameters
        'rate_drop': trial.suggest_float('rate_drop', 0.0, 0.5),
        'skip_drop': trial.suggest_float('skip_drop', 0.0, 0.5),
        
        'random_state': 42
    }
    
    # Train model
    model = xgb.XGBClassifier(**param)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Predict on validation set
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate score based on specified metric
    if metric == 'accuracy':
        score = accuracy_score(y_val, y_pred)
    elif metric == 'roc_auc':
        score = roc_auc_score(y_val, y_pred_proba)
    elif metric == 'f1':
        score = f1_score(y_val, y_pred)
    elif metric == 'log_loss':
        score = -log_loss(y_val, y_pred_proba)  # Negative because Optuna maximizes
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return score

def perform_bayesian_optimization(X_train, y_train, X_val, y_val, n_trials=100, metric='accuracy', study_name=None, storage=None):
    """
    Perform Bayesian hyperparameter optimization using Optuna
    
    Parameters:
    - X_train: Training features
    - y_train: Training target
    - X_val: Validation features
    - y_val: Validation target
    - n_trials: Number of optimization trials
    - metric: Metric to optimize
    - study_name: Optional name for the Optuna study
    - storage: Optional storage URL for the Optuna study
    
    Returns:
    - best_params: Dictionary of best parameters
    - study: Optuna study object
    """
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=storage,
        load_if_exists=True
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective_xgboost(trial, X_train, y_train, X_val, y_val, metric),
        n_trials=n_trials
    )
    
    # Get best parameters
    best_params = study.best_params
    
    # Add fixed parameters
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = 'logloss'
    best_params['random_state'] = 42
    
    return best_params, study

def plot_optimization_history(study, output_dir=None):
    """
    Plot optimization history
    
    Parameters:
    - study: Optuna study object
    - output_dir: Optional directory to save plots
    """
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Plot optimization history
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'optimization_history.png'))
        plt.close()
    else:
        plt.show()
    
    # Plot parameter importances
    plt.figure(figsize=(10, 8))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'param_importances.png'))
        plt.close()
    else:
        plt.show()
    
    # Plot parallel coordinate plot
    plt.figure(figsize=(12, 8))
    optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'parallel_coordinate.png'))
        plt.close()
    else:
        plt.show()

def time_based_hyperparameter_tuning(X, y, preprocessor, n_splits=5, n_trials=50, metric='accuracy', output_dir=None):
    """
    Perform hyperparameter tuning with time-based cross-validation
    
    Parameters:
    - X: Feature matrix
    - y: Target vector
    - preprocessor: Feature preprocessor
    - n_splits: Number of time series splits
    - n_trials: Number of optimization trials per split
    - metric: Metric to optimize
    - output_dir: Optional directory to save results
    
    Returns:
    - best_params: Dictionary of best parameters
    - cv_results: DataFrame with cross-validation results
    """
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create time series split
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Prepare results storage
    cv_results = []
    best_params_per_fold = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"Tuning fold {fold+1}/{n_splits}")
        
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Further split train into train and validation
        train_size = int(0.8 * len(X_train))
        X_train_inner, X_val = X_train.iloc[:train_size], X_train.iloc[train_size:]
        y_train_inner, y_val = y_train.iloc[:train_size], y_train.iloc[train_size:]
        
        # Preprocess data
        X_train_inner_transformed = preprocessor.fit_transform(X_train_inner)
        X_val_transformed = preprocessor.transform(X_val)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Optimize hyperparameters
        fold_output_dir = os.path.join(output_dir, f"fold_{fold+1}") if output_dir else None
        
        best_params, study = perform_bayesian_optimization(
            X_train_inner_transformed, y_train_inner,
            X_val_transformed, y_val,
            n_trials=n_trials,
            metric=metric,
            study_name=f"fold_{fold+1}"
        )
        
        # Train model with best parameters
        model = xgb.XGBClassifier(**best_params)
        model.fit(
            X_train_inner_transformed, y_train_inner,
            eval_set=[(X_val_transformed, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Evaluate on test set
        y_pred = model.predict(X_test_transformed)
        y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        loss = log_loss(y_test, y_pred_proba)
        
        # Record results
        result = {
            'fold': fold + 1,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'f1': f1,
            'log_loss': loss,
            'n_train': len(X_train_inner),
            'n_val': len(X_val),
            'n_test': len(X_test)
        }
        
        cv_results.append(result)
        best_params_per_fold.append(best_params)
        
        # Plot optimization results
        if fold_output_dir:
            os.makedirs(fold_output_dir, exist_ok=True)
            plot_optimization_history(study, fold_output_dir)
            
            # Save best parameters
            with open(os.path.join(fold_output_dir, 'best_params.txt'), 'w') as f:
                for param, value in best_params.items():
                    f.write(f"{param}: {value}\n")
            
            # Save fold metrics
            pd.DataFrame([result]).to_csv(os.path.join(fold_output_dir, 'metrics.csv'), index=False)
    
    # Convert results to DataFrame
    cv_results_df = pd.DataFrame(cv_results)
    
    # Aggregate cross-validation results
    mean_results = cv_results_df.mean().to_dict()
    std_results = cv_results_df.std().to_dict()
    
    print("\n=== Cross-Validation Results ===")
    print(f"Mean Accuracy: {mean_results['accuracy']:.4f} ± {std_results['accuracy']:.4f}")
    print(f"Mean ROC AUC: {mean_results['roc_auc']:.4f} ± {std_results['roc_auc']:.4f}")
    print(f"Mean F1 Score: {mean_results['f1']:.4f} ± {std_results['f1']:.4f}")
    print(f"Mean Log Loss: {mean_results['log_loss']:.4f} ± {std_results['log_loss']:.4f}")
    
    # Find best parameters across all folds
    best_fold_idx = cv_results_df[metric].idxmax()
    best_params = best_params_per_fold[best_fold_idx]
    
    print(f"\nBest parameters from fold {best_fold_idx + 1}:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Save overall results
    if output_dir:
        # Save cross-validation results
        cv_results_df.to_csv(os.path.join(output_dir, 'cv_results.csv'), index=False)
        
        # Save overall metrics
        overall_metrics = {
            'mean_accuracy': mean_results['accuracy'],
            'std_accuracy': std_results['accuracy'],
            'mean_roc_auc': mean_results['roc_auc'],
            'std_roc_auc': std_results['roc_auc'],
            'mean_f1': mean_results['f1'],
            'std_f1': std_results['f1'],
            'mean_log_loss': mean_results['log_loss'],
            'std_log_loss': std_results['log_loss'],
            'best_fold': int(best_fold_idx + 1)
        }
        
        pd.DataFrame([overall_metrics]).to_csv(os.path.join(output_dir, 'overall_metrics.csv'), index=False)
        
        # Save best parameters
        with open(os.path.join(output_dir, 'best_params.txt'), 'w') as f:
            f.write(f"Best parameters from fold {best_fold_idx + 1}:\n")
            for param, value in best_params.items():
                f.write(f"{param}: {value}\n")
        
        # Save best parameters as joblib
        joblib.dump(best_params, os.path.join(output_dir, 'best_params.pkl'))
        
        # Save timestamp
        with open(os.path.join(output_dir, 'tuning_timestamp.txt'), 'w') as f:
            f.write(f"Tuning performed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return best_params, cv_results_df

def train_model_with_best_params(X_train, y_train, X_val, y_val, best_params, preprocessor=None):
    """
    Train model with best hyperparameters
    
    Parameters:
    - X_train: Training features
    - y_train: Training target
    - X_val: Validation features
    - y_val: Validation target
    - best_params: Dictionary of best parameters
    - preprocessor: Optional feature preprocessor
    
    Returns:
    - model: Trained model
    """
    # Preprocess data if preprocessor is provided
    if preprocessor:
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_val_transformed = preprocessor.transform(X_val)
    else:
        X_train_transformed = X_train
        X_val_transformed = X_val
    
    # Train model
    model = xgb.XGBClassifier(**best_params)
    model.fit(
        X_train_transformed, y_train,
        eval_set=[(X_val_transformed, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    return model 