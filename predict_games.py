import pandas as pd
import numpy as np
import joblib
import os
import argparse
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from nba_data_fetcher import fetch_upcoming_schedule, fetch_team_stats

def load_model_and_preprocessor(model_dir='model'):
    """
    Load the trained model and preprocessor
    
    Parameters:
    - model_dir: Directory containing model files
    
    Returns:
    - model: Trained XGBoost model
    - preprocessor: Fitted preprocessor
    - feature_columns: List of feature columns
    - categorical_features: List of categorical features
    """
    model = joblib.load(os.path.join(model_dir, 'xgboost_model.pkl'))
    preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.pkl'))
    feature_columns = joblib.load(os.path.join(model_dir, 'feature_columns.pkl'))
    categorical_features = joblib.load(os.path.join(model_dir, 'categorical_features.pkl'))
    
    return model, preprocessor, feature_columns, categorical_features

def predict_upcoming_games(upcoming_games_data, model, preprocessor, feature_columns, categorical_features):
    """
    Predict outcomes for upcoming games
    
    Parameters:
    - upcoming_games_data: DataFrame with upcoming games
    - model: Trained model
    - preprocessor: Fitted preprocessor
    - feature_columns: List of feature columns
    - categorical_features: List of categorical features
    
    Returns:
    - predictions_df: DataFrame with game predictions
    """
    # Prepare features for prediction
    X = upcoming_games_data[feature_columns + categorical_features]
    
    # Transform features
    X_transformed = preprocessor.transform(X)
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_transformed)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Create results DataFrame
    predictions_df = pd.DataFrame({
        'home_team': upcoming_games_data['team_abbreviation_home'],
        'away_team': upcoming_games_data['team_abbreviation_away'],
        'game_date': upcoming_games_data['game_date'],
        'home_win_probability': y_pred_proba,
        'away_win_probability': 1 - y_pred_proba,
        'predicted_winner': np.where(y_pred == 1, 
                                    upcoming_games_data['team_abbreviation_home'], 
                                    upcoming_games_data['team_abbreviation_away']),
        'prediction_confidence': np.maximum(y_pred_proba, 1 - y_pred_proba)
    })
    
    return predictions_df

def create_matchup_data(home_team, away_team, date=None, team_stats_file=None, custom_stats=None):
    """
    Create feature data for a matchup between two teams
    
    Parameters:
    - home_team: Home team abbreviation
    - away_team: Away team abbreviation
    - date: Game date (default: today)
    - team_stats_file: Optional file with team stats
    - custom_stats: Optional dictionary with custom stats
    
    Returns:
    - matchup_df: DataFrame with game features
    """
    if date is None:
        date = datetime.now()
    elif isinstance(date, str):
        date = pd.to_datetime(date)
    
    # Initialize matchup dictionary
    matchup = {
        'team_abbreviation_home': home_team,
        'team_abbreviation_away': away_team,
        'game_date': date,
        'dayofweek': date.dayofweek,
        'month': date.month
    }
    
    # If team stats file is provided, load team stats
    if team_stats_file and os.path.exists(team_stats_file):
        team_stats = pd.read_csv(team_stats_file)
        
        # Get home team stats
        home_stats = team_stats[team_stats['team_abbreviation'] == home_team].iloc[0].to_dict() if len(team_stats[team_stats['team_abbreviation'] == home_team]) > 0 else {}
        
        # Get away team stats
        away_stats = team_stats[team_stats['team_abbreviation'] == away_team].iloc[0].to_dict() if len(team_stats[team_stats['team_abbreviation'] == away_team]) > 0 else {}
        
        # Add team stats to matchup
        for stat, value in home_stats.items():
            if stat not in ['team_abbreviation', 'team_name']:
                matchup[f'{stat}_home'] = value
        
        for stat, value in away_stats.items():
            if stat not in ['team_abbreviation', 'team_name']:
                matchup[f'{stat}_away'] = value
        
        # Calculate differentials
        for stat in home_stats:
            if stat not in ['team_abbreviation', 'team_name'] and stat in away_stats:
                matchup[f'{stat}_diff'] = home_stats[stat] - away_stats[stat]
    
    # If custom stats are provided, add them
    if custom_stats:
        for key, value in custom_stats.items():
            matchup[key] = value
    
    # Create DataFrame
    matchup_df = pd.DataFrame([matchup])
    
    return matchup_df

def create_upcoming_schedule(schedule_file, team_stats_file=None):
    """
    Create prediction data for an upcoming schedule
    
    Parameters:
    - schedule_file: CSV file with upcoming schedule
    - team_stats_file: Optional file with team stats
    
    Returns:
    - upcoming_games_df: DataFrame with upcoming games data
    """
    # Load schedule
    schedule = pd.read_csv(schedule_file)
    
    # Ensure date is in datetime format
    schedule['game_date'] = pd.to_datetime(schedule['game_date'])
    
    # Initialize list for game data
    games_data = []
    
    # Process each game
    for _, game in schedule.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']
        game_date = game['game_date']
        
        # Create matchup data
        matchup_data = create_matchup_data(home_team, away_team, game_date, team_stats_file)
        
        # Add to list
        games_data.append(matchup_data)
    
    # Combine all games
    upcoming_games_df = pd.concat(games_data, ignore_index=True)
    
    return upcoming_games_df

def visualize_predictions(predictions_df, save_path=None):
    """
    Visualize game predictions
    
    Parameters:
    - predictions_df: DataFrame with game predictions
    - save_path: Optional path to save the visualization
    """
    # Sort by game date
    predictions_df = predictions_df.sort_values('game_date')
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create barplot for win probabilities
    sns.barplot(x='home_team', y='home_win_probability', data=predictions_df, 
                color='blue', label='Home Win Probability')
    
    # Add game information
    for i, row in enumerate(predictions_df.itertuples()):
        plt.text(i, 0.05, f"{row.away_team}", ha='center', color='white', fontweight='bold')
        plt.text(i, row.home_win_probability + 0.03, f"{row.home_win_probability:.2f}", ha='center')
        plt.text(i, row.home_win_probability - 0.08, f"{1-row.home_win_probability:.2f}", ha='center', color='white')
        
        # Highlight predicted winner
        winner_color = 'blue' if row.predicted_winner == row.home_team else 'red'
        plt.text(i, 1.03, f"{row.predicted_winner}", ha='center', color=winner_color, fontweight='bold')
    
    # Set labels and title
    plt.xlabel('Home Team')
    plt.ylabel('Win Probability')
    plt.title('NBA Game Predictions')
    plt.ylim(0, 1.1)
    
    # Format x-axis labels
    plt.xticks(rotation=45)
    
    # Add legend
    plt.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def predict_single_game(home_team, away_team, game_date=None, team_stats_file=None, model_dir='model'):
    """
    Predict the outcome of a single game
    
    Parameters:
    - home_team: Home team abbreviation
    - away_team: Away team abbreviation
    - game_date: Game date (default: today)
    - team_stats_file: Optional file with team stats
    - model_dir: Directory containing model files
    
    Returns:
    - result: Dictionary with prediction results
    """
    # Load model and preprocessor
    model, preprocessor, feature_columns, categorical_features = load_model_and_preprocessor(model_dir)
    
    # Create matchup data
    matchup_df = create_matchup_data(home_team, away_team, game_date, team_stats_file)
    
    # Make prediction
    predictions_df = predict_upcoming_games(matchup_df, model, preprocessor, feature_columns, categorical_features)
    
    # Get prediction
    result = predictions_df.iloc[0].to_dict()
    
    return result

def predict_and_save_upcoming_games(schedule_file, team_stats_file=None, model_dir='model', output_dir='predictions'):
    """
    Predict and save outcomes for upcoming games
    
    Parameters:
    - schedule_file: CSV file with upcoming schedule
    - team_stats_file: Optional file with team stats
    - model_dir: Directory containing model files
    - output_dir: Directory to save prediction results
    
    Returns:
    - predictions_df: DataFrame with game predictions
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and preprocessor
    model, preprocessor, feature_columns, categorical_features = load_model_and_preprocessor(model_dir)
    
    # Create upcoming schedule data
    upcoming_games_df = create_upcoming_schedule(schedule_file, team_stats_file)
    
    # Make predictions
    predictions_df = predict_upcoming_games(upcoming_games_df, model, preprocessor, feature_columns, categorical_features)
    
    # Save predictions
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    predictions_df.to_csv(os.path.join(output_dir, f'predictions_{timestamp}.csv'), index=False)
    
    # Create visualization
    visualize_predictions(predictions_df, save_path=os.path.join(output_dir, f'predictions_viz_{timestamp}.png'))
    
    return predictions_df

def main():
    """
    Main function to predict NBA games
    """
    parser = argparse.ArgumentParser(description='Predict NBA games')
    parser.add_argument('--model_dir', default='model', help='Directory containing model files')
    parser.add_argument('--output_dir', default='predictions', help='Directory to save predictions')
    parser.add_argument('--single_game', action='store_true', help='Predict a single game')
    parser.add_argument('--home_team', help='Home team abbreviation for single game prediction')
    parser.add_argument('--away_team', help='Away team abbreviation for single game prediction')
    parser.add_argument('--game_date', help='Game date for single game prediction (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.single_game:
        if not args.home_team or not args.away_team:
            print("Error: --home_team and --away_team are required for single game prediction")
            return
        
        # Fetch team stats
        team_stats = fetch_team_stats()
        team_stats.to_csv('data/team_stats.csv', index=False)
        
        # Predict single game
        result = predict_single_game(
            args.home_team,
            args.away_team,
            args.game_date,
            'data/team_stats.csv',
            args.model_dir
        )
        
        # Print prediction
        print("\nGame Prediction:")
        print(f"{args.home_team} vs {args.away_team}")
        print(f"Date: {result['game_date']}")
        print(f"Predicted Winner: {result['predicted_winner']}")
        print(f"Home Win Probability: {result['home_win_probability']:.2%}")
        print(f"Away Win Probability: {result['away_win_probability']:.2%}")
        print(f"Prediction Confidence: {result['prediction_confidence']:.2%}")
        
    else:
        # Fetch upcoming schedule and team stats
        print("Fetching upcoming schedule and team stats...")
        upcoming_schedule = fetch_upcoming_schedule()
        team_stats = fetch_team_stats()
        
        if upcoming_schedule.empty:
            print("No upcoming games found in the next 7 days.")
            return
        
        # Create prediction data
        upcoming_games_df = create_upcoming_schedule(
            'data/upcoming_schedule.csv',
            'data/team_stats.csv'
        )
        
        # Load model and preprocessor
        model, preprocessor, feature_columns, categorical_features = load_model_and_preprocessor(args.model_dir)
        
        # Make predictions
        predictions_df = predict_upcoming_games(
            upcoming_games_df,
            model,
            preprocessor,
            feature_columns,
            categorical_features
        )
        
        # Save predictions
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        predictions_file = os.path.join(args.output_dir, f'predictions_{timestamp}.csv')
        predictions_df.to_csv(predictions_file, index=False)
        
        # Create visualization
        viz_file = os.path.join(args.output_dir, f'predictions_{timestamp}.png')
        visualize_predictions(predictions_df, viz_file)
        
        print(f"\nPredictions saved to {predictions_file}")
        print(f"Visualization saved to {viz_file}")

if __name__ == "__main__":
    main() 