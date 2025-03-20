# NBA Game Predictor

A machine learning model that predicts NBA game outcomes using historical data and team statistics.

## Features

- Fetches historical game data and team statistics directly from the NBA API
- Automatically retrieves upcoming schedule for the next 7 days
- Predicts game outcomes using an XGBoost model
- Provides win probabilities and confidence scores
- Generates visualizations of predictions
- Supports both single game and batch predictions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ML-NBA-Game-Predictor.git
cd ML-NBA-Game-Predictor
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Fetch NBA Data

To fetch the latest NBA data (historical games, upcoming schedule, and team statistics):

```bash
python nba_data_fetcher.py
```

This will create a `data` directory containing:
- `historical_games.csv`: Historical game data from the last 3 seasons
- `upcoming_schedule.csv`: Upcoming games for the next 7 days
- `team_stats.csv`: Team statistics and averages

### Predict Single Game

To predict the outcome of a single game:

```bash
python predict_games.py --single_game --home_team LAL --away_team BOS --game_date 2024-03-25
```

### Predict Upcoming Games

To predict outcomes for all upcoming games in the next 7 days:

```bash
python predict_games.py
```

The predictions will be saved in the `predictions` directory with timestamps:
- `predictions_YYYYMMDD_HHMMSS.csv`: Detailed predictions in CSV format
- `predictions_YYYYMMDD_HHMMSS.png`: Visualization of predictions

### Additional Options

- `--model_dir`: Specify the directory containing model files (default: 'model')
- `--output_dir`: Specify the directory to save predictions (default: 'predictions')

## Model Details

The model uses the following features:
- Team statistics (points, field goal percentage, rebounds, etc.)
- Home/Away team indicators
- Game date features (day of week, month)
- Historical performance metrics

## Data Sources

All data is fetched directly from the official NBA API using the `nba_api` package. The data includes:
- Game results and box scores
- Team statistics and averages
- Current season schedule
- Team information and abbreviations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 