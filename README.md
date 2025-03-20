# NBA Game Prediction Project

A machine learning system that predicts NBA basketball game outcomes using historical game data and XGBoost classification.

## Overview

This project uses advanced machine learning techniques to predict the winners of NBA games. The system:

1. Processes historical NBA game data
2. Performs extensive feature engineering
3. Trains an XGBoost model with Bayesian hyperparameter optimization
4. Evaluates model performance with time-series cross-validation
5. Provides tools to make predictions on new games

## Features

- **Advanced Feature Engineering**: Team performance metrics, rest days analysis, matchup history, and winning/losing streaks
- **Bayesian Hyperparameter Optimization**: Uses Optuna to find optimal model parameters
- **Time Series Validation**: Ensures the model is evaluated on future games
- **Model Explainability**: SHAP values to explain predictions
- **Prediction Interface**: Command-line tools to predict single games or entire schedules

## Project Structure

- `nba_game_predictor.py`: Main module for loading data, feature engineering, and model training
- `advanced_features.py`: Module for creating advanced team and game features
- `model_evaluation.py`: Functions for evaluating model performance
- `hyperparameter_tuning.py`: Bayesian optimization for XGBoost hyperparameters
- `predict_games.py`: Command-line interface for predicting games

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/nba-game-predictor.git
cd nba-game-predictor
```

2. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

### Training a Model

```python
python nba_game_predictor.py
```

This will:
1. Load historical game data
2. Engineer features
3. Train a model with optimized hyperparameters
4. Evaluate performance
5. Save the model to the `model` directory

### Predicting a Single Game

```bash
python predict_games.py single --home LAL --away BOS --stats team_stats.csv
```

### Predicting Games from a Schedule

```bash
python predict_games.py schedule --schedule upcoming_games.csv --stats team_stats.csv
```

## Data Format

### Game Data

The model expects NBA game data in CSV format with the following columns:
- `game_date`: Date of the game
- `team_abbreviation_home`: Home team abbreviation (e.g., 'LAL')
- `team_abbreviation_away`: Away team abbreviation (e.g., 'BOS')
- Various team stats columns with '_home' and '_away' suffixes

### Upcoming Schedule

The prediction script expects a CSV file with:
- `game_date`: Date of the game
- `home_team`: Home team abbreviation
- `away_team`: Away team abbreviation

## Model Performance

The model achieves:
- Accuracy: ~70% on test data
- ROC AUC: ~0.75
- Log Loss: ~0.55

## Future Improvements

- Incorporate player-level data and injury information
- Add betting odds as features
- Explore ensemble methods combining multiple models
- Develop a web interface for predictions

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 