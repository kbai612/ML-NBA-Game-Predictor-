import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    log_loss, brier_score_loss
)
from sklearn.model_selection import TimeSeriesSplit, KFold
import xgboost as xgb
import shap
import os
from datetime import datetime
import joblib

def evaluate_predictions(y_true, y_pred, y_pred_proba):
    """
    Evaluate binary classification predictions with multiple metrics
    
    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - y_pred_proba: Predicted probabilities for the positive class
    
    Returns:
    - Dictionary with evaluation metrics
    """
    metrics = {}
    
    # Classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred)
    
    # Probability metrics
    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    metrics['log_loss'] = log_loss(y_true, y_pred_proba)
    metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
    
    return metrics

def print_evaluation_report(metrics):
    """
    Print a formatted evaluation report
    
    Parameters:
    - metrics: Dictionary with evaluation metrics
    """
    print("\n=== Model Evaluation Report ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Log Loss: {metrics['log_loss']:.4f}")
    print(f"Brier Score: {metrics['brier_score']:.4f}")
    print("===============================\n")

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix
    
    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - save_path: Path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot ROC curve
    
    Parameters:
    - y_true: True labels
    - y_pred_proba: Predicted probabilities for the positive class
    - save_path: Path to save the figure
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_calibration_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot calibration curve
    
    Parameters:
    - y_true: True labels
    - y_pred_proba: Predicted probabilities for the positive class
    - save_path: Path to save the figure
    """
    from sklearn.calibration import calibration_curve
    
    plt.figure(figsize=(8, 6))
    
    # Plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    
    # Plot model calibration
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """
    Plot feature importance
    
    Parameters:
    - model: Trained XGBoost model
    - feature_names: List of feature names
    - top_n: Number of top features to display
    - save_path: Path to save the figure
    """
    # Get feature importance from the model
    importance = model.feature_importances_
    
    # Create a dataframe for visualization
    feat_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    # Select top N features
    feat_importance = feat_importance.head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_importance)
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
    return feat_importance

def plot_shap_summary(model, X, feature_names=None, save_path=None):
    """
    Plot SHAP summary
    
    Parameters:
    - model: Trained model
    - X: Feature matrix
    - feature_names: List of feature names
    - save_path: Path to save the figure
    """
    # Create SHAP explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title('SHAP Feature Importance')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def time_series_cross_validation(X, y, model_class, param_grid, n_splits=5, scoring='accuracy'):
    """
    Perform time series cross-validation
    
    Parameters:
    - X: Feature matrix
    - y: Target vector
    - model_class: Model class to use
    - param_grid: Parameter grid for hyperparameter tuning
    - n_splits: Number of time series splits
    - scoring: Scoring metric
    
    Returns:
    - Best model
    - Cross-validation results
    """
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    
    # Create time series split
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Setup grid search with time series CV
    grid_search = GridSearchCV(
        estimator=model_class(),
        param_grid=param_grid,
        cv=tscv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X, y)
    
    # Get results
    cv_results = pd.DataFrame(grid_search.cv_results_)
    
    return grid_search.best_estimator_, cv_results

def evaluate_with_walk_forward_validation(X, y, model, preprocessor, test_size=0.2, step_size=0.05, min_train_size=0.5):
    """
    Evaluate model with walk-forward validation (expanding window)
    
    Parameters:
    - X: Feature matrix
    - y: Target vector
    - model: Model to evaluate
    - preprocessor: Feature preprocessor
    - test_size: Size of the test set at each step
    - step_size: Step size for each validation window
    - min_train_size: Minimum size of training data
    
    Returns:
    - Dictionary with performance metrics for each validation window
    """
    total_samples = len(X)
    evaluation_results = []
    
    # Calculate number of evaluation steps
    n_steps = int((1 - min_train_size - test_size) / step_size) + 1
    
    for i in range(n_steps):
        # Calculate split points
        train_end = int((min_train_size + i * step_size) * total_samples)
        test_start = train_end
        test_end = test_start + int(test_size * total_samples)
        
        # Ensure we don't go beyond the available data
        if test_end > total_samples:
            break
        
        # Split data
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]
        
        # Preprocess data
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Train and predict
        model.fit(X_train_transformed, y_train)
        y_pred = model.predict(X_test_transformed)
        y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]
        
        # Evaluate
        metrics = evaluate_predictions(y_test, y_pred, y_pred_proba)
        
        # Record results
        metrics['fold'] = i + 1
        metrics['train_size'] = len(X_train)
        metrics['test_size'] = len(X_test)
        metrics['train_start_date'] = X.index[0]
        metrics['train_end_date'] = X.index[train_end - 1] if train_end > 0 else None
        metrics['test_start_date'] = X.index[test_start] if test_start < total_samples else None
        metrics['test_end_date'] = X.index[test_end - 1] if test_end - 1 < total_samples else None
        
        evaluation_results.append(metrics)
    
    return pd.DataFrame(evaluation_results)

def backtesting_evaluation(X, y, model, preprocessor, initial_train_size=0.5, step_size=50):
    """
    Evaluate model with backtesting (expanding window, adding samples incrementally)
    
    Parameters:
    - X: Feature matrix
    - y: Target vector
    - model: Model to evaluate
    - preprocessor: Feature preprocessor
    - initial_train_size: Initial size of training data as a fraction
    - step_size: Number of samples to add in each step
    
    Returns:
    - DataFrame with performance metrics for each test period
    """
    total_samples = len(X)
    initial_train_end = int(initial_train_size * total_samples)
    
    results = []
    
    for test_start in range(initial_train_end, total_samples, step_size):
        test_end = min(test_start + step_size, total_samples)
        
        # Split data
        X_train = X.iloc[:test_start]
        y_train = y.iloc[:test_start]
        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]
        
        # Skip if test set is empty
        if len(X_test) == 0:
            continue
        
        # Preprocess data
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Train and predict
        model.fit(X_train_transformed, y_train)
        y_pred = model.predict(X_test_transformed)
        y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]
        
        # Skip if no positive samples in test set
        if len(np.unique(y_test)) < 2:
            continue
        
        # Evaluate
        metrics = evaluate_predictions(y_test, y_pred, y_pred_proba)
        
        # Record results
        metrics['test_start'] = test_start
        metrics['test_end'] = test_end
        metrics['test_size'] = len(X_test)
        
        results.append(metrics)
    
    return pd.DataFrame(results)

def evaluate_model_and_save_results(model, X_test, y_test, preprocessor, feature_names, output_dir):
    """
    Evaluate model on test set and save results
    
    Parameters:
    - model: Trained model
    - X_test: Test features
    - y_test: Test targets
    - preprocessor: Feature preprocessor
    - feature_names: List of feature names
    - output_dir: Directory to save results
    
    Returns:
    - Dictionary with evaluation metrics
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Preprocess test data
    X_test_transformed = preprocessor.transform(X_test)
    
    # Generate predictions
    y_pred = model.predict(X_test_transformed)
    y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]
    
    # Evaluate predictions
    metrics = evaluate_predictions(y_test, y_pred, y_pred_proba)
    
    # Print evaluation report
    print_evaluation_report(metrics)
    
    # Plot and save confusion matrix
    plot_confusion_matrix(y_test, y_pred, save_path=os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Plot and save ROC curve
    plot_roc_curve(y_test, y_pred_proba, save_path=os.path.join(output_dir, 'roc_curve.png'))
    
    # Plot and save calibration curve
    plot_calibration_curve(y_test, y_pred_proba, save_path=os.path.join(output_dir, 'calibration_curve.png'))
    
    # Plot and save feature importance
    importance_df = plot_feature_importance(model, feature_names, save_path=os.path.join(output_dir, 'feature_importance.png'))
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    # Plot and save SHAP summary
    plot_shap_summary(model, X_test_transformed, feature_names, save_path=os.path.join(output_dir, 'shap_summary.png'))
    
    # Save full classification report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(classification_report(y_test, y_pred))
    
    # Save metrics to CSV
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, 'evaluation_metrics.csv'), index=False)
    
    # Save predictions
    pd.DataFrame({
        'true_label': y_test,
        'predicted_label': y_pred,
        'predicted_probability': y_pred_proba
    }).to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    
    # Save timestamp
    with open(os.path.join(output_dir, 'evaluation_timestamp.txt'), 'w') as f:
        f.write(f"Evaluation performed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return metrics 