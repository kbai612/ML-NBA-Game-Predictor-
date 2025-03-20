import pandas as pd
import numpy as np
from tqdm import tqdm

def create_team_features(df, window_sizes=[5, 10, 20]):
    """
    Create advanced team features based on historical performance
    
    Parameters:
    - df: DataFrame with game data
    - window_sizes: List of window sizes for rolling averages
    
    Returns:
    - DataFrame with additional team features
    """
    print("Creating advanced team features...")
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_values('game_date')
    
    # Create team stats dictionaries
    team_stats = {}
    
    # List of basic stats to track
    basic_stats = [
        'fg_pct', 'fg3_pct', 'ft_pct', 'ast', 'reb', 
        'stl', 'blk', 'tov', 'pts'
    ]
    
    # Initialize team tracking dataframes
    for team in pd.concat([data['team_abbreviation_home'], data['team_abbreviation_away']]).unique():
        team_stats[team] = pd.DataFrame()
    
    # Process games in chronological order
    print("Processing games to create team performance metrics...")
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        game_date = row['game_date']
        home_team = row['team_abbreviation_home']
        away_team = row['team_abbreviation_away']
        
        # Record home team stats
        game_stats_home = {}
        game_stats_home['game_date'] = game_date
        game_stats_home['opponent'] = away_team
        game_stats_home['is_home'] = 1
        game_stats_home['win'] = 1 if row['home_win'] == 1 else 0
        
        for stat in basic_stats:
            game_stats_home[stat] = row[f'{stat}_home']
        
        # Record away team stats
        game_stats_away = {}
        game_stats_away['game_date'] = game_date
        game_stats_away['opponent'] = home_team
        game_stats_away['is_home'] = 0
        game_stats_away['win'] = 0 if row['home_win'] == 1 else 1
        
        for stat in basic_stats:
            game_stats_away[stat] = row[f'{stat}_away']
        
        # Append to team stats
        team_stats[home_team] = pd.concat([team_stats[home_team], pd.DataFrame([game_stats_home])], ignore_index=True)
        team_stats[away_team] = pd.concat([team_stats[away_team], pd.DataFrame([game_stats_away])], ignore_index=True)
    
    # Calculate historical stats for each team
    print("Calculating rolling averages for teams...")
    for team, stats_df in team_stats.items():
        if len(stats_df) > 0:
            stats_df = stats_df.sort_values('game_date')
            
            # Calculate rolling win percentage
            stats_df['win_pct'] = stats_df['win'].expanding().mean()
            
            # Calculate home/away win percentage
            home_games = stats_df[stats_df['is_home'] == 1]
            away_games = stats_df[stats_df['is_home'] == 0]
            
            if len(home_games) > 0:
                stats_df['home_win_pct'] = stats_df['is_home'] * home_games['win'].expanding().mean()
            else:
                stats_df['home_win_pct'] = 0
                
            if len(away_games) > 0:
                stats_df['away_win_pct'] = (1 - stats_df['is_home']) * away_games['win'].expanding().mean()
            else:
                stats_df['away_win_pct'] = 0
            
            # Calculate rolling averages for basic stats
            for stat in basic_stats:
                for window in window_sizes:
                    # Only calculate if we have enough games
                    if len(stats_df) >= window:
                        stats_df[f'{stat}_last_{window}'] = stats_df[stat].rolling(window=window, min_periods=1).mean()
                    else:
                        stats_df[f'{stat}_last_{window}'] = stats_df[stat].expanding().mean()
            
            # Calculate momentum (weighted average of recent games)
            # More weight to recent games
            stats_df['momentum_score'] = 0
            if len(stats_df) >= 10:
                weights = np.array([0.25, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03])
                for i in range(10, len(stats_df) + 1):
                    last_10_wins = stats_df['win'].iloc[i-10:i].values
                    stats_df.loc[stats_df.index[i-1], 'momentum_score'] = np.sum(last_10_wins * weights)
            
            # Update the team stats dictionary
            team_stats[team] = stats_df
    
    # Add features to the original dataframe
    print("Adding team features to the main dataframe...")
    features_home = []
    features_away = []
    
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        game_date = row['game_date']
        home_team = row['team_abbreviation_home']
        away_team = row['team_abbreviation_away']
        
        # Get home team stats before this game
        home_team_stats = team_stats[home_team]
        home_team_past = home_team_stats[home_team_stats['game_date'] < game_date]
        
        # Get away team stats before this game
        away_team_stats = team_stats[away_team]
        away_team_past = away_team_stats[away_team_stats['game_date'] < game_date]
        
        # Create feature dictionaries
        home_features = {}
        away_features = {}
        
        # Recent performance metrics
        if len(home_team_past) > 0:
            home_features['win_pct'] = home_team_past['win_pct'].iloc[-1]
            home_features['home_win_pct'] = home_team_past['home_win_pct'].iloc[-1]
            home_features['momentum_score'] = home_team_past['momentum_score'].iloc[-1]
            
            # Add rolling averages
            for stat in basic_stats:
                for window in window_sizes:
                    if f'{stat}_last_{window}' in home_team_past.columns:
                        home_features[f'{stat}_last_{window}'] = home_team_past[f'{stat}_last_{window}'].iloc[-1]
        else:
            # If no past games, use default values
            home_features['win_pct'] = 0.5
            home_features['home_win_pct'] = 0.5
            home_features['momentum_score'] = 0.5
            
            for stat in basic_stats:
                for window in window_sizes:
                    home_features[f'{stat}_last_{window}'] = 0
        
        if len(away_team_past) > 0:
            away_features['win_pct'] = away_team_past['win_pct'].iloc[-1]
            away_features['away_win_pct'] = away_team_past['away_win_pct'].iloc[-1]
            away_features['momentum_score'] = away_team_past['momentum_score'].iloc[-1]
            
            # Add rolling averages
            for stat in basic_stats:
                for window in window_sizes:
                    if f'{stat}_last_{window}' in away_team_past.columns:
                        away_features[f'{stat}_last_{window}'] = away_team_past[f'{stat}_last_{window}'].iloc[-1]
        else:
            # If no past games, use default values
            away_features['win_pct'] = 0.5
            away_features['away_win_pct'] = 0.5
            away_features['momentum_score'] = 0.5
            
            for stat in basic_stats:
                for window in window_sizes:
                    away_features[f'{stat}_last_{window}'] = 0
        
        features_home.append(home_features)
        features_away.append(away_features)
    
    # Convert lists to dataframes
    home_features_df = pd.DataFrame(features_home)
    away_features_df = pd.DataFrame(features_away)
    
    # Add prefixes to column names
    home_features_df.columns = ['home_' + col for col in home_features_df.columns]
    away_features_df.columns = ['away_' + col for col in away_features_df.columns]
    
    # Add features to original dataframe
    data = pd.concat([data.reset_index(drop=True), 
                      home_features_df.reset_index(drop=True), 
                      away_features_df.reset_index(drop=True)], axis=1)
    
    return data

def calculate_rest_days(df):
    """
    Calculate the number of rest days for each team
    
    Parameters:
    - df: DataFrame with game data
    
    Returns:
    - DataFrame with rest days features
    """
    print("Calculating rest days for each team...")
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_values('game_date')
    
    # Create dictionaries to track last game date for each team
    last_game_date = {}
    
    # Initialize rest days columns
    data['rest_days_home'] = 0
    data['rest_days_away'] = 0
    
    # Process games in chronological order
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        game_date = row['game_date']
        home_team = row['team_abbreviation_home']
        away_team = row['team_abbreviation_away']
        
        # Calculate rest days for home team
        if home_team in last_game_date:
            data.at[idx, 'rest_days_home'] = (game_date - last_game_date[home_team]).days
        else:
            data.at[idx, 'rest_days_home'] = 3  # Default value for first game
        
        # Calculate rest days for away team
        if away_team in last_game_date:
            data.at[idx, 'rest_days_away'] = (game_date - last_game_date[away_team]).days
        else:
            data.at[idx, 'rest_days_away'] = 3  # Default value for first game
        
        # Update last game date
        last_game_date[home_team] = game_date
        last_game_date[away_team] = game_date
    
    # Calculate rest advantage
    data['rest_advantage'] = data['rest_days_home'] - data['rest_days_away']
    
    # Create categorical features for back-to-back games
    data['home_b2b'] = (data['rest_days_home'] == 0).astype(int)
    data['away_b2b'] = (data['rest_days_away'] == 0).astype(int)
    
    return data

def create_matchup_features(df):
    """
    Create features based on historical matchups between teams
    
    Parameters:
    - df: DataFrame with game data
    
    Returns:
    - DataFrame with matchup features
    """
    print("Creating matchup-specific features...")
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_values('game_date')
    
    # Create a dictionary to store matchup history
    matchup_history = {}
    
    # Initialize matchup columns
    data['matchup_home_win_pct'] = 0.5  # Default to 0.5 for no prior history
    data['matchup_last_N_wins'] = 0.5   # Default to 0.5 for no prior history
    
    # Process games in chronological order
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        home_team = row['team_abbreviation_home']
        away_team = row['team_abbreviation_away']
        matchup_key = f"{home_team}_{away_team}"
        
        # Get current matchup history
        if matchup_key not in matchup_history:
            matchup_history[matchup_key] = []
        
        # Calculate matchup features
        if len(matchup_history[matchup_key]) > 0:
            # Overall win percentage of home team against this opponent
            home_wins = sum(matchup_history[matchup_key])
            total_games = len(matchup_history[matchup_key])
            data.at[idx, 'matchup_home_win_pct'] = home_wins / total_games
            
            # Recent matchup history (last 3 games or all if fewer)
            n_games = min(3, len(matchup_history[matchup_key]))
            recent_wins = sum(matchup_history[matchup_key][-n_games:])
            data.at[idx, 'matchup_last_N_wins'] = recent_wins / n_games
        
        # Update matchup history after the game
        home_win = row['home_win']
        matchup_history[matchup_key].append(home_win)
        
        # Also track the reverse matchup (when teams play at the other venue)
        reverse_key = f"{away_team}_{home_team}"
        if reverse_key not in matchup_history:
            matchup_history[reverse_key] = []
        # For the reverse matchup, a home win is an away loss
        matchup_history[reverse_key].append(1 - home_win)
    
    return data

def create_streak_features(df):
    """
    Create features related to winning and losing streaks
    
    Parameters:
    - df: DataFrame with game data
    
    Returns:
    - DataFrame with streak features
    """
    print("Creating streak features...")
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_values('game_date')
    
    # Create dictionaries to track streaks for each team
    win_streak = {}
    loss_streak = {}
    
    # Initialize streak columns
    data['home_win_streak'] = 0
    data['home_loss_streak'] = 0
    data['away_win_streak'] = 0
    data['away_loss_streak'] = 0
    
    # Process games in chronological order
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        home_team = row['team_abbreviation_home']
        away_team = row['team_abbreviation_away']
        
        # Initialize if first appearance
        if home_team not in win_streak:
            win_streak[home_team] = 0
            loss_streak[home_team] = 0
        
        if away_team not in win_streak:
            win_streak[away_team] = 0
            loss_streak[away_team] = 0
        
        # Record current streaks
        data.at[idx, 'home_win_streak'] = win_streak[home_team]
        data.at[idx, 'home_loss_streak'] = loss_streak[home_team]
        data.at[idx, 'away_win_streak'] = win_streak[away_team]
        data.at[idx, 'away_loss_streak'] = loss_streak[away_team]
        
        # Update streaks based on game outcome
        home_win = row['home_win']
        
        # Update home team streaks
        if home_win == 1:
            win_streak[home_team] += 1
            loss_streak[home_team] = 0
        else:
            win_streak[home_team] = 0
            loss_streak[home_team] += 1
        
        # Update away team streaks
        if home_win == 0:
            win_streak[away_team] += 1
            loss_streak[away_team] = 0
        else:
            win_streak[away_team] = 0
            loss_streak[away_team] += 1
    
    return data

def create_advanced_features(df):
    """
    Create all advanced features for the NBA prediction model
    
    Parameters:
    - df: DataFrame with game data
    
    Returns:
    - DataFrame with all advanced features
    """
    # Create team performance features
    data = create_team_features(df)
    
    # Calculate rest days
    data = calculate_rest_days(data)
    
    # Create matchup features
    data = create_matchup_features(data)
    
    # Create streak features
    data = create_streak_features(data)
    
    print("Advanced feature engineering completed")
    return data 