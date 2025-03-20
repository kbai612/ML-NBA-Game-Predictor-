# transform_nba_stats.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round, expr, when
import argparse

def main():
    parser = argparse.ArgumentParser(description='Transform NBA stats data')
    parser.add_argument('--input', required=True, help='Input S3 path')
    parser.add_argument('--output', required=True, help='Output S3 path')
    args = parser.parse_args()
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("NBA Stats Transformation") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .getOrCreate()
    
    # Read data from S3
    df = spark.read.csv(args.input, header=True, inferSchema=True)
    
    # Perform transformations
    transformed_df = df.select(
        col("PLAYER_ID").alias("player_id"),
        col("PLAYER_NAME").alias("player_name"),
        col("SEASON_ID").alias("season_id"),
        col("TEAM_ID").alias("team_id"),
        col("TEAM_ABBREVIATION").alias("team_abbreviation"),
        col("PLAYER_AGE").alias("player_age"),
        col("GP").alias("gp"),
        round(col("PTS"), 2).alias("pts"),
        round(col("REB"), 2).alias("reb"),
        round(col("AST"), 2).alias("ast"),
        round(col("FG_PCT"), 4).alias("fg_pct"),
        round(col("FG3_PCT"), 4).alias("fg3_pct"),
        round(col("FT_PCT"), 4).alias("ft_pct"),
        round(col("MIN"), 2).alias("min")
    )
    
    # Calculate efficiency rating (PER-like simplified metric)
    transformed_df = transformed_df.withColumn(
        "efficiency_rating",
        round(
            (col("pts") + col("reb") + col("ast") * 1.5 + 
             col("fg_pct") * 100 * 0.5 + col("ft_pct") * 100 * 0.5) / col("gp"),
            2
        )
    )
    
    # Filter out rows with null values in important columns
    final_df = transformed_df.filter(
        col("pts").isNotNull() & 
        col("gp").isNotNull() & 
        col("player_name").isNotNull()
    )
    
    # Write transformed data to S3 in Parquet format
    final_df.write.mode("overwrite").parquet(args.output)
    
    spark.stop()

if __name__ == "__main__":
    main()
