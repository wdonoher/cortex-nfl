"""
Cortex Sports Analytics — NFL Automated Pipeline
=================================================
The unattended job that runs on Railway's scheduler. Fetches everything
via API, computes Elo/EPA/Unit Ratings, writes to Postgres, syncs to S3.
No manual data handling anywhere in this file — if a step ever requires
someone to fetch or upload something by hand, that step is a bug.

FILE ORGANIZATION (assumed repo layout):
    nfl_feature_engineering.py   <- NFLEloEngine, EPAFeatureEngine, UnitRatingEngine
    nfl_pipeline.py               <- this file, imports the above

ENVIRONMENT VARIABLES (set in Railway's project settings, not in code):
    DATABASE_URL           - Postgres connection string (Railway sets this
                              automatically for linked Postgres services)
    S3_BUCKET               - optional, for raw/feature snapshot exports
    AWS_ACCESS_KEY_ID       - optional, required only if S3_BUCKET is set
    AWS_SECRET_ACCESS_KEY   - optional, required only if S3_BUCKET is set
    AWS_REGION               - optional, defaults to 'us-east-1'

RAILWAY CRON SETUP:
    Weekly job (Thursday 10:00 AM ET, matching the agreed pipeline cadence):
        schedule: "0 14 * * 4"   (14:00 UTC ≈ 10:00 AM ET)
        command:  python nfl_pipeline.py --mode weekly

    Historical backfill (run ONCE via Railway's one-off "Run Command",
    not on the recurring cron schedule):
        command:  python nfl_pipeline.py --mode backfill --start-year 2018 --end-year 2025

Both modes are fully scripted — the backfill is a one-off *job invocation*,
not a manual data-handling step. Nothing is downloaded to a computer,
copied, or uploaded by a person at any point.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from io import BytesIO

import pandas as pd
import numpy as np
import nfl_data_py as nfl
from sqlalchemy import create_engine, text

from nfl_feature_engineering import NFLEloEngine, EPAFeatureEngine, UnitRatingEngine

try:
    import boto3
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger('nfl_pipeline')


# ============================================================================
# SEASON HELPERS
# ============================================================================

def determine_current_season(today=None):
    """
    NFL seasons are labeled by the year they start (e.g. games from
    Sept 2026 through Feb 2027 are all 'season 2026'). Jan/Feb games
    belong to the *previous* label.
    """
    today = today or datetime.utcnow()
    return today.year if today.month >= 3 else today.year - 1


# ============================================================================
# PIPELINE
# ============================================================================

class NFLPipeline:

    def __init__(self, database_url=None, s3_bucket=None):
        self.database_url = database_url or os.environ.get('DATABASE_URL')
        self.s3_bucket = s3_bucket or os.environ.get('S3_BUCKET')

        if not self.database_url:
            raise RuntimeError(
                "DATABASE_URL not set. This must be configured in Railway's "
                "environment variables — never hardcoded or passed manually."
            )

        self.engine = create_engine(self.database_url)

        self.s3_client = None
        if self.s3_bucket:
            if not S3_AVAILABLE:
                logger.warning("S3_BUCKET set but boto3 not installed; skipping S3 sync.")
            else:
                self.s3_client = boto3.client(
                    's3', region_name=os.environ.get('AWS_REGION', 'us-east-1')
                )

    # ------------------------------------------------------------------
    # SOURCE DATA (all pulled via API — nfl_data_py -> nflverse GitHub)
    # ------------------------------------------------------------------

    def fetch_source_data(self, seasons):
        logger.info(f"Fetching source data for seasons: {seasons}")

        schedule = nfl.import_schedules(seasons)
        logger.info(f"  schedule: {len(schedule)} rows")

        pbp = nfl.import_pbp_data(seasons, downcast=True)
        logger.info(f"  play-by-play: {len(pbp)} rows")

        try:
            injuries = nfl.import_injuries(seasons)
            logger.info(f"  injuries: {len(injuries)} rows")
        except Exception as e:
            logger.warning(f"  injuries fetch failed, continuing without: {e}")
            injuries = pd.DataFrame()

        try:
            rosters = nfl.import_rosters(seasons)
            logger.info(f"  rosters: {len(rosters)} rows")
        except Exception as e:
            logger.warning(f"  rosters fetch failed, continuing without: {e}")
            rosters = pd.DataFrame()

        return {
            'schedule': schedule,
            'pbp': pbp,
            'injuries': injuries,
            'rosters': rosters,
        }

    # ------------------------------------------------------------------
    # SHAPE DATA FOR THE FEATURE ENGINES
    # ------------------------------------------------------------------

    def build_games_df(self, schedule_df):
        """Completed games only, in the schema NFLEloEngine expects."""
        games = schedule_df[schedule_df['home_score'].notna()].copy()
        games = games.rename(columns={'game_id': 'game_id'})
        return games[['game_id', 'season', 'week', 'home_team', 'away_team',
                       'home_score', 'away_score']]

    def build_schedule_unrolled(self, schedule_df):
        """
        One row per team per game (not per game), with an 'opponent'
        column — the shape UnitRatingEngine's opponent-adjustment needs.
        """
        played = schedule_df[schedule_df['home_score'].notna()].copy()

        home_rows = played[['season', 'week', 'home_team', 'away_team']].rename(
            columns={'home_team': 'team', 'away_team': 'opponent'}
        )
        away_rows = played[['season', 'week', 'home_team', 'away_team']].rename(
            columns={'away_team': 'team', 'home_team': 'opponent'}
        )

        return pd.concat([home_rows, away_rows], ignore_index=True)

    # ------------------------------------------------------------------
    # FEATURE ENGINEERING
    # ------------------------------------------------------------------

    def run_feature_engineering(self, games_df, pbp_df, schedule_unrolled_df):
        logger.info("Running Elo engine...")
        elo_engine = NFLEloEngine()
        elo_history = elo_engine.process_games(games_df)

        logger.info("Running EPA feature engine...")
        epa_engine = EPAFeatureEngine(window=4)
        epa_features = epa_engine.compute(pbp_df)

        logger.info("Running Tier 1 unit rating engine...")
        unit_engine = UnitRatingEngine(window=4)
        unit_ratings = unit_engine.compute_unit_ratings(pbp_df, schedule_unrolled_df)

        return elo_history, epa_features, unit_ratings

    # ------------------------------------------------------------------
    # DATABASE
    # ------------------------------------------------------------------

    def ensure_tables(self):
        """Creates feature tables if they don't already exist. Safe to
        run every time — CREATE TABLE IF NOT EXISTS is idempotent."""
        ddl = """
        CREATE TABLE IF NOT EXISTS nfl_elo_ratings (
            game_id TEXT PRIMARY KEY,
            season INTEGER,
            week INTEGER,
            home_team TEXT,
            away_team TEXT,
            home_rating_pre DOUBLE PRECISION,
            away_rating_pre DOUBLE PRECISION,
            home_rating_post DOUBLE PRECISION,
            away_rating_post DOUBLE PRECISION,
            home_win_prob_pre DOUBLE PRECISION,
            updated_at TIMESTAMP DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS nfl_epa_features (
            team TEXT,
            season INTEGER,
            week INTEGER,
            epa_off_play DOUBLE PRECISION,
            epa_off_play_roll4 DOUBLE PRECISION,
            epa_off_play_trend_slope DOUBLE PRECISION,
            epa_def_play_allowed DOUBLE PRECISION,
            epa_def_play_allowed_roll4 DOUBLE PRECISION,
            epa_def_play_allowed_trend_slope DOUBLE PRECISION,
            updated_at TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (team, season, week)
        );

        CREATE TABLE IF NOT EXISTS nfl_unit_ratings (
            team TEXT,
            season INTEGER,
            week INTEGER,
            pass_protection_rating DOUBLE PRECISION,
            pass_protection_trend DOUBLE PRECISION,
            pass_rush_rating DOUBLE PRECISION,
            pass_rush_trend DOUBLE PRECISION,
            passing_offense_rating DOUBLE PRECISION,
            passing_offense_trend DOUBLE PRECISION,
            pass_defense_rating DOUBLE PRECISION,
            pass_defense_trend DOUBLE PRECISION,
            run_blocking_rating DOUBLE PRECISION,
            run_blocking_trend DOUBLE PRECISION,
            run_defense_rating DOUBLE PRECISION,
            run_defense_trend DOUBLE PRECISION,
            updated_at TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (team, season, week)
        );
        """
        with self.engine.begin() as conn:
            for statement in ddl.strip().split(';'):
                if statement.strip():
                    conn.execute(text(statement))
        logger.info("Tables verified/created.")

    def upsert_by_keys(self, df, table_name, key_cols):
        """
        Idempotent write: deletes any existing rows matching the key
        combinations present in df, then inserts. Safe to re-run the
        same week (or a full backfill) repeatedly without duplicating
        or erroring on conflict.
        """
        if df.empty:
            logger.info(f"  {table_name}: no rows to write, skipping.")
            return

        df = df.replace({np.nan: None})

        with self.engine.begin() as conn:
            key_tuples = df[key_cols].drop_duplicates()

            for _, row in key_tuples.iterrows():
                conditions = " AND ".join([f'"{k}" = :{k}' for k in key_cols])
                params = {k: row[k] for k in key_cols}
                conn.execute(
                    text(f'DELETE FROM {table_name} WHERE {conditions}'),
                    params
                )

            df.to_sql(table_name, conn, if_exists='append', index=False)

        logger.info(f"  {table_name}: wrote {len(df)} rows.")

    # ------------------------------------------------------------------
    # S3 SYNC (optional snapshot export)
    # ------------------------------------------------------------------

    def sync_to_s3(self, df, key):
        if not self.s3_client:
            return
        buffer = BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        self.s3_client.put_object(Bucket=self.s3_bucket, Key=key, Body=buffer.getvalue())
        logger.info(f"  synced to s3://{self.s3_bucket}/{key}")

    # ------------------------------------------------------------------
    # ENTRY POINTS
    # ------------------------------------------------------------------

    def run(self, seasons, run_label):
        source = self.fetch_source_data(seasons)

        games_df = self.build_games_df(source['schedule'])
        schedule_unrolled = self.build_schedule_unrolled(source['schedule'])

        elo_history, epa_features, unit_ratings = self.run_feature_engineering(
            games_df, source['pbp'], schedule_unrolled
        )

        self.ensure_tables()

        logger.info("Writing to Postgres...")
        self.upsert_by_keys(elo_history, 'nfl_elo_ratings', ['game_id'])
        self.upsert_by_keys(epa_features, 'nfl_epa_features', ['team', 'season', 'week'])
        self.upsert_by_keys(unit_ratings, 'nfl_unit_ratings', ['team', 'season', 'week'])

        if self.s3_client:
            logger.info("Syncing snapshots to S3...")
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            self.sync_to_s3(elo_history, f'nfl/{run_label}/elo_{timestamp}.parquet')
            self.sync_to_s3(epa_features, f'nfl/{run_label}/epa_{timestamp}.parquet')
            self.sync_to_s3(unit_ratings, f'nfl/{run_label}/unit_ratings_{timestamp}.parquet')

        logger.info(f"Pipeline run complete ({run_label}).")

    def run_weekly(self):
        season = determine_current_season()
        logger.info(f"=== WEEKLY RUN — season {season} ===")
        self.run(seasons=[season], run_label='weekly')

    def run_backfill(self, start_year, end_year):
        seasons = list(range(start_year, end_year + 1))
        logger.info(f"=== BACKFILL RUN — seasons {seasons} ===")
        self.run(seasons=seasons, run_label='backfill')


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Cortex Sports Analytics — NFL Pipeline")
    parser.add_argument('--mode', choices=['weekly', 'backfill'], required=True)
    parser.add_argument('--start-year', type=int, default=2018)
    parser.add_argument('--end-year', type=int, default=None)
    args = parser.parse_args()

    pipeline = NFLPipeline()

    if args.mode == 'weekly':
        pipeline.run_weekly()
    else:
        end_year = args.end_year or determine_current_season()
        pipeline.run_backfill(args.start_year, end_year)


if __name__ == "__main__":
    main()
