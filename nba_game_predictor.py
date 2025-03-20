import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
import shap
import os
import optuna
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_data(filepath):
    """
    Load NBA game data from CSV file
    """
    df = pd.read_csv(filepath)
    
    # Filter to regular season games from 2010 onwards
    df = df[(df['game_date'] >= '2010-01-01') & (df['season_type'] == 'Regular Season')]
    
    # Create target variable: 1 if home team wins, 0 if away team wins
    df['home_win'] = (df['pts_home'] > df['pts_away']).astype(int)
    
    # Convert game_date to datetime
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    # Sort by date
    df = df.sort_values('game_date')
    
    print(f"Loaded {len(df)} games from {df['game_date'].min()} to {df['game_date'].max()}")
    
    return df

def engineer_features(df):
    """
    Create features for the model
    """
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Extract features like day of week, month, etc.
    data['dayofweek'] = data['game_date'].dt.dayofweek
    data['month'] = data['game_date'].dt.month
    data['season'] = data['game_date'].dt.year
    
    # Calculate point differentials
    data['point_diff'] = data['pts_home'] - data['pts_away']
    
    # Create shooting efficiency metrics
    data['fg_pct_diff'] = data['fg_pct_home'] - data['fg_pct_away']
    data['fg3_pct_diff'] = data['fg3_pct_home'] - data['fg3_pct_away']
    data['ft_pct_diff'] = data['ft_pct_home'] - data['ft_pct_away']
    
    # Create defensive metrics
    data['stl_diff'] = data['stl_home'] - data['stl_away']
    data['blk_diff'] = data['blk_home'] - data['blk_away']
    data['reb_diff'] = data['reb_home'] - data['reb_away']
    data['oreb_diff'] = data['oreb_home'] - data['oreb_away']
    data['dreb_diff'] = data['dreb_home'] - data['dreb_away']
    data['ast_diff'] = data['ast_home'] - data['ast_away']
    data['tov_diff'] = data['tov_home'] - data['tov_away']
    
    # Create team strength features (rolling averages)
    # This requires additional historical data to implement correctly
    
    return data

def prepare_train_test_sets(data, test_size=0.2, val_size=0.1):
    """
    Prepare train, validation and test sets
    Using time series split - training on earlier games, testing on later games
    """
    # Define features and target
    # Drop game outcome related columns and irrelevant columns
    drop_columns = [
        'home_win', 'wl_home', 'wl_away', 'pts_home', 'pts_away', 
        'game_id', 'team_id_home', 'team_id_away', 'team_name_home', 'team_name_away',
        'matchup_home', 'matchup_away', 'plus_minus_home', 'plus_minus_away',
        'video_available_home', 'video_available_away', 'point_diff'
    ]
    
    # Convert categorical features
    categorical_features = ['team_abbreviation_home', 'team_abbreviation_away', 'dayofweek', 'month']
    
    # Get temporal features for cross-validation
    temporal_features = ['game_date', 'season_id']
    
    # Get all numeric features
    feature_columns = [col for col in data.columns if col not in drop_columns + categorical_features + temporal_features]
    
    # Split data chronologically
    train_val_data = data.iloc[:-int(len(data) * test_size)]
    test_data = data.iloc[-int(len(data) * test_size):]
    
    # Further split train into train and validation
    train_data = train_val_data.iloc[:-int(len(train_val_data) * val_size/(1-test_size))]
    val_data = train_val_data.iloc[-int(len(train_val_data) * val_size/(1-test_size)):]
    
    print(f"Train data: {len(train_data)} games from {train_data['game_date'].min()} to {train_data['game_date'].max()}")
    print(f"Validation data: {len(val_data)} games from {val_data['game_date'].min()} to {val_data['game_date'].max()}")
    print(f"Test data: {len(test_data)} games from {test_data['game_date'].min()} to {test_data['game_date'].max()}")
    
    # Prepare feature preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, feature_columns),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Prepare train features and target
    X_train = train_data[feature_columns + categorical_features]
    y_train = train_data['home_win']
    
    # Prepare validation features and target
    X_val = val_data[feature_columns + categorical_features]
    y_val = val_data['home_win']
    
    # Prepare test features and target
    X_test = test_data[feature_columns + categorical_features]
    y_test = test_data['home_win']
    
    return X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, feature_columns, categorical_features

def objective(trial, X_train, y_train, X_val, y_val, preprocessor):
    """
    Optuna objective function for hyperparameter tuning
    """
    # Create preprocessing pipeline with XGBoost
    param = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': RANDOM_STATE
    }
    
    # Fit preprocessing on training data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    
    # Use early stopping to prevent overfitting
    model = xgb.XGBClassifier(**param)
    model.fit(
        X_train_transformed, y_train,
        eval_set=[(X_val_transformed, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Evaluate on validation set
    preds = model.predict(X_val_transformed)
    accuracy = accuracy_score(y_val, preds)
    
    return accuracy

def train_model(X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, feature_columns, categorical_features):
    """
    Train XGBoost model with optimized hyperparameters
    """
    # Optimize hyperparameters with Optuna
    print("Optimizing hyperparameters with Bayesian optimization...")
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, preprocessor),
        n_trials=20  # Increase for better results
    )
    
    best_params = study.best_params
    print(f"Best parameters: {best_params}")
    
    # Create final model with best parameters
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = 'logloss'
    best_params['random_state'] = RANDOM_STATE
    
    # Create the final pipeline
    model = xgb.XGBClassifier(**best_params)
    
    # Fit preprocessing on combined train and validation data
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    
    X_train_val_transformed = preprocessor.fit_transform(X_train_val)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Train the final model
    print("Training final model...")
    model.fit(X_train_val_transformed, y_train_val)
    
    # Evaluate on test set
    y_pred = model.predict(X_test_transformed)
    y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test precision: {precision:.4f}")
    print(f"Test recall: {recall:.4f}")
    print(f"Test F1 score: {f1:.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    
    # Feature importance
    feature_names = feature_columns + list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features))
    
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(model, max_num_features=20)
    plt.title('Feature Importance')
    plt.savefig('feature_importance.png')
    
    # SHAP values for explainability
    print("Calculating SHAP values for model explainability...")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test_transformed)
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names)
    plt.savefig('shap_summary.png')
    
    # Save model and preprocessor
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(model_dir, 'xgboost_model.pkl'))
    joblib.dump(preprocessor, os.path.join(model_dir, 'preprocessor.pkl'))
    joblib.dump(feature_columns, os.path.join(model_dir, 'feature_columns.pkl'))
    joblib.dump(categorical_features, os.path.join(model_dir, 'categorical_features.pkl'))
    
    print(f"Model and preprocessor saved to {model_dir}")
    
    return model, preprocessor

def predict_game(model, preprocessor, feature_columns, categorical_features, home_team, away_team, home_stats, away_stats):
    """
    Predict the outcome of a game with the trained model
    
    Parameters:
    - model: Trained XGBoost model
    - preprocessor: Fitted preprocessor
    - home_team: Home team abbreviation (e.g., 'LAL')
    - away_team: Away team abbreviation (e.g., 'BOS')
    - home_stats: Dictionary with home team stats
    - away_stats: Dictionary with away team stats
    
    Returns:
    - win_probability: Probability of home team winning
    """
    # Create a dataframe with the game features
    game_features = {}
    
    # Add team abbreviations
    game_features['team_abbreviation_home'] = home_team
    game_features['team_abbreviation_away'] = away_team
    
    # Add game date features
    today = datetime.now()
    game_features['dayofweek'] = today.dayofweek
    game_features['month'] = today.month
    
    # Add team stats
    for stat, value in home_stats.items():
        game_features[f'{stat}_home'] = value
    
    for stat, value in away_stats.items():
        game_features[f'{stat}_away'] = value
    
    # Calculate differentials
    for stat in home_stats:
        if stat in away_stats:
            game_features[f'{stat}_diff'] = home_stats[stat] - away_stats[stat]
    
    # Create dataframe
    game_df = pd.DataFrame([game_features])
    
    # Select relevant features
    game_features_df = game_df[feature_columns + categorical_features]
    
    # Apply preprocessing
    game_features_transformed = preprocessor.transform(game_features_df)
    
    # Predict
    win_probability = model.predict_proba(game_features_transformed)[0, 1]
    
    return win_probability

def main():
    """
    Main function to run the NBA game prediction pipeline
    """
    # Load data
    data_path = input("Enter path to game.csv file: ")
    df = load_data(data_path)
    
    # Engineer features
    data = engineer_features(df)
    
    # Prepare train, validation and test sets
    X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, feature_columns, categorical_features = prepare_train_test_sets(data)
    
    # Train and evaluate model
    model, preprocessor = train_model(X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, feature_columns, categorical_features)
    
    print("\nModel training complete! The model can now be used to predict NBA game outcomes.")
    print("To use the model for prediction, load it with joblib and use the predict_game function.")

if __name__ == "__main__":
    main() 