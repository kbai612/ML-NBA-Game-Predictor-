import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamefinder, scoreboardv2, teamgamelog
from nba_api.stats.static import teams
from datetime import datetime, timedelta
import time
import os

def get_team_abbreviations():
    """Get dictionary mapping team IDs to abbreviations"""
    nba_teams = teams.get_teams()
    return {team['id']: team['abbreviation'] for team in nba_teams}

def fetch_historical_games(seasons=None):
    """
    Fetch historical game data from NBA API
    
    Parameters:
    - seasons: List of seasons to fetch (e.g., ['2022-23', '2021-22'])
    
    Returns:
    - DataFrame with historical game data
    """
    if seasons is None:
        current_year = datetime.now().year
        seasons = [f"{year}-{str(year + 1)[-2:]}" for year in range(current_year - 3, current_year)]
    
    all_games = []
    team_abbreviations = get_team_abbreviations()
    
    for season in seasons:
        print(f"Fetching games for season {season}...")
        
        # Get all games for the season
        gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
        games = gamefinder.get_data_frames()[0]
        
        # Process games
        games['SEASON'] = season
        games['HOME_TEAM'] = games['TEAM_ID'].map(team_abbreviations)
        games['AWAY_TEAM'] = games['OPPONENT_TEAM_ID'].map(team_abbreviations)
        
        all_games.append(games)
        time.sleep(1)  # Rate limiting
    
    # Combine all seasons
    historical_games = pd.concat(all_games, ignore_index=True)
    
    # Save to CSV
    historical_games.to_csv('data/historical_games.csv', index=False)
    return historical_games

def fetch_upcoming_schedule():
    """
    Fetch upcoming schedule from NBA API
    
    Returns:
    - DataFrame with upcoming games
    """
    # Get today's date
    today = datetime.now()
    
    # Get schedule for next 7 days
    upcoming_games = []
    for i in range(7):
        game_date = today + timedelta(days=i)
        date_str = game_date.strftime('%Y-%m-%d')
        
        try:
            scoreboard = scoreboardv2.ScoreboardV2(game_date=date_str)
            games = scoreboard.get_data_frames()[0]
            
            if not games.empty:
                games['GAME_DATE'] = date_str
                upcoming_games.append(games)
            
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            print(f"Error fetching games for {date_str}: {str(e)}")
    
    if not upcoming_games:
        return pd.DataFrame()
    
    # Combine all upcoming games
    upcoming_schedule = pd.concat(upcoming_games, ignore_index=True)
    
    # Process and clean the data
    team_abbreviations = get_team_abbreviations()
    upcoming_schedule['HOME_TEAM'] = upcoming_schedule['HOME_TEAM_ID'].map(team_abbreviations)
    upcoming_schedule['AWAY_TEAM'] = upcoming_schedule['VISITOR_TEAM_ID'].map(team_abbreviations)
    
    # Select and rename relevant columns
    upcoming_schedule = upcoming_schedule[[
        'GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'HOME_TEAM_NAME', 'VISITOR_TEAM_NAME'
    ]].rename(columns={
        'GAME_DATE': 'game_date',
        'HOME_TEAM': 'home_team',
        'AWAY_TEAM': 'away_team',
        'HOME_TEAM_NAME': 'home_team_name',
        'VISITOR_TEAM_NAME': 'away_team_name'
    })
    
    # Save to CSV
    upcoming_schedule.to_csv('data/upcoming_schedule.csv', index=False)
    return upcoming_schedule

def fetch_team_stats(seasons=None):
    """
    Fetch team statistics from NBA API
    
    Parameters:
    - seasons: List of seasons to fetch (e.g., ['2022-23', '2021-22'])
    
    Returns:
    - DataFrame with team statistics
    """
    if seasons is None:
        current_year = datetime.now().year
        seasons = [f"{year}-{str(year + 1)[-2:]}" for year in range(current_year - 3, current_year)]
    
    all_stats = []
    team_abbreviations = get_team_abbreviations()
    
    for season in seasons:
        print(f"Fetching team stats for season {season}...")
        
        # Get all teams
        nba_teams = teams.get_teams()
        
        for team in nba_teams:
            try:
                # Get team game log
                gamelog = teamgamelog.TeamGameLog(team_id=team['id'], season=season)
                stats = gamelog.get_data_frames()[0]
                
                if not stats.empty:
                    stats['SEASON'] = season
                    stats['TEAM_ABBREVIATION'] = team['abbreviation']
                    all_stats.append(stats)
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"Error fetching stats for {team['abbreviation']} in {season}: {str(e)}")
    
    if not all_stats:
        return pd.DataFrame()
    
    # Combine all team stats
    team_stats = pd.concat(all_stats, ignore_index=True)
    
    # Calculate team averages
    team_averages = team_stats.groupby(['TEAM_ABBREVIATION', 'SEASON']).agg({
        'PTS': 'mean',
        'FG_PCT': 'mean',
        'FG3_PCT': 'mean',
        'FT_PCT': 'mean',
        'REB': 'mean',
        'AST': 'mean',
        'STL': 'mean',
        'BLK': 'mean',
        'TOV': 'mean'
    }).reset_index()
    
    # Save to CSV
    team_averages.to_csv('data/team_stats.csv', index=False)
    return team_averages

def main():
    """Main function to fetch and process all NBA data"""
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Fetch all data
    print("Fetching historical games...")
    historical_games = fetch_historical_games()
    
    print("\nFetching upcoming schedule...")
    upcoming_schedule = fetch_upcoming_schedule()
    
    print("\nFetching team statistics...")
    team_stats = fetch_team_stats()
    
    print("\nData fetching complete! Files saved in the 'data' directory:")
    print("- historical_games.csv")
    print("- upcoming_schedule.csv")
    print("- team_stats.csv")

if __name__ == "__main__":
    main() 