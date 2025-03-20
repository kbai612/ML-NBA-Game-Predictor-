from airflow import DAG
from airflow.decorators import task
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.amazon.aws.operators.redshift import RedshiftSQLOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

from datetime import datetime, timedelta
import pandas as pd
import boto3
import os
import json
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import players

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 2, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    'nba_stats_etl_pipeline',
    default_args=default_args,
    description='ETL pipeline for NBA stats data',
    schedule_interval='@daily',
    catchup=False
) as dag:

    @task
    def extract_nba_data(**kwargs):
        """Extract player career stats from the NBA API"""
        # Get all active players
        all_players = players.get_active_players()
        
        # For demo purposes, limit to 10 players
        player_sample = all_players[:10]
        
        # Extract career stats for each player
        player_stats = []
        for player in player_sample:
            player_id = player['id']
            player_name = player['full_name']
            
            print(f"Extracting stats for {player_name}")
            
            # Get career stats
            career = playercareerstats.PlayerCareerStats(player_id=player_id)
            career_data = career.get_data_frames()[0]  # Regular season stats
            
            # Add player name for easier identification
            career_data['PLAYER_NAME'] = player_name
            
            player_stats.append(career_data)
        
        # Combine all player stats
        combined_stats = pd.concat(player_stats, ignore_index=True)
        
        # Save to local temp file
        temp_file_path = '/tmp/nba_player_stats.csv'
        combined_stats.to_csv(temp_file_path, index=False)
        
        # Upload to S3
        s3_hook = S3Hook(aws_conn_id='aws_default')
        s3_hook.load_file(
            filename=temp_file_path,
            key=f'raw/nba_stats/{kwargs["ds"]}/player_stats.csv',
            bucket_name='nba-stats-etl-bucket',
            replace=True
        )
        
        return f's3://nba-stats-etl-bucket/raw/nba_stats/{kwargs["ds"]}/player_stats.csv'

    # PySpark transformation job
    transform_data = SparkSubmitOperator(
        task_id='transform_data_with_pyspark',
        application='{{ dag_run.conf["SPARK_SCRIPT_PATH"] }}/transform_nba_stats.py',
        conn_id='spark_default',
        application_args=[
            '--input', '{{ task_instance.xcom_pull(task_ids="extract_nba_data") }}',
            '--output', 's3://nba-stats-etl-bucket/transformed/nba_stats/{{ ds }}/'
        ],
        conf={
            'spark.hadoop.fs.s3a.aws.credentials.provider': 'org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider'
        }
    )
    
    # Create Redshift table if not exists
    create_table = RedshiftSQLOperator(
        task_id='create_redshift_table',
        redshift_conn_id='redshift_default',
        sql="""
        CREATE TABLE IF NOT EXISTS nba_player_stats (
            player_id INTEGER,
            player_name VARCHAR(255),
            season_id VARCHAR(20),
            team_id INTEGER,
            team_abbreviation VARCHAR(10),
            player_age INTEGER,
            gp INTEGER,
            pts DECIMAL(10,2),
            reb DECIMAL(10,2),
            ast DECIMAL(10,2),
            fg_pct DECIMAL(5,4),
            fg3_pct DECIMAL(5,4),
            ft_pct DECIMAL(5,4),
            min DECIMAL(10,2),
            efficiency_rating DECIMAL(10,2)
        )
        """
    )
    
    # Load data into Redshift
    load_to_redshift = RedshiftSQLOperator(
        task_id='load_to_redshift',
        redshift_conn_id='redshift_default',
        sql="""
        COPY nba_player_stats
        FROM 's3://nba-stats-etl-bucket/transformed/nba_stats/{{ ds }}/part-*'
        IAM_ROLE 'arn:aws:iam::123456789012:role/RedshiftS3Role'
        FORMAT AS PARQUET;
        """
    )
    
    # Define task dependencies
    extract_task = extract_nba_data()
    extract_task >> transform_data >> create_table >> load_to_redshift
