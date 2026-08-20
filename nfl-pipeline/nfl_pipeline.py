"""
Cortex Sports Analytics — NFL Automated Pipeline
=================================================
The unattended job that runs on Railway's scheduler. Fetches everything
via API, computes Elo/EPA/Unit Ratings, writes to Postgres, syncs to S3.

MIGRATED to nflreadpy (from deprecated nfl_data_py) — nflreadpy is the
actively maintained official successor. Returns Polars DataFrames,
converted to pandas immediately after each fetch since everything
downstream (feature engines, projection model) is pandas-based.

NEW (Aug 2026): after the main weekly update, also generates forward-
looking win-probability predictions for the upcoming week's games,
using NFLEloEngine.predict_upcoming_games() against current ratings.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from io import BytesIO

import pandas as pd
import numpy as np
import nflreadpy as nfl
from sqlalchemy import create_engine, text

from nfl_feature_engineering import NFLEloEngine, EPAFeatureEngine, UnitRatingEngine

try:
    import boto3
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger('nfl_pipeline')

PBP_COLUMNS = [
    'game_id', 'season', 'week', 'posteam', 'defteam',
    'epa', 'pass_attempt', 'rush_attempt', 'sack', 'qb_hit',
]


def determine_current_season(today=None):
    today = today or datetime.utcnow()
    return today.year if today.month >= 3 else today.year - 1


def _downcast_floats(df):
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    return df


def send_failure_alert(run_label, error, seasons=None):
    webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
    if not webhook_url:
        logger.warning(
            "SLACK_WEBHOOK_URL not set — failure alert not sent. "
            "This run's failure will only be visible in Railway logs."
        )
        return
    if not REQUESTS_AVAILABLE:
        logger.warning("requests library not installed — cannot send Slack alert.")
        return

    season_note = f" (seasons: {seasons})" if seasons else ""
    payload = {
        "text": (
            f":rotating_light: *Cortex NFL Pipeline Failure*\n"
            f"Run: `{run_label}`{season_note}\n"
            f"Error: `{str(error)}`\n"
            f"Time (UTC): {datetime.utcnow().isoformat()}"
        )
    }

    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Failure alert sent to Slack.")
    except Exception as alert_err:
        logger.error(f"Failed to send Slack alert: {alert_err}")


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

    def fetch_source_data(self, seasons):
        logger.info(f"Fetching source data for seasons: {seasons}")

        schedule = nfl.load_schedules(seasons).to_pandas()
        logger.info(f"  schedule: {len(schedule)} rows")

        pbp_pl = nfl.load_pbp(seasons)
        available_pbp_cols = [c for c in PBP_COLUMNS if c in pbp_pl.columns]
        pbp = pbp_pl.select(available_pbp_cols).to_pandas()
        pbp = _downcast_floats(pbp)
        logger.info(f"  play-by-play: {len(pbp)} rows, {len(pbp.columns)} columns")

        try:
            injuries = nfl.load_injuries(seasons).to_pandas()
            logger.info(f"  injuries: {len(injuries)} rows")
        except Exception as e:
            logger.warning(f"  injuries fetch failed, continuing without: {e}")
            injuries = pd.DataFrame()

        try:
            rosters = nfl.load_rosters(seasons).to_pandas()
            logger.info(f"  rosters: {len(rosters)} rows")
        except Exception as e:
            logger.warning(f"  rosters fetch failed, continuing without: {e}")
            rosters = pd.DataFrame()

        try:
            weekly_stats = self._fetch_weekly_stats_per_season(seasons)
            logger.info(f"  weekly player stats: {len(weekly_stats)} rows")
        except Exception as e:
            logger.warning(f"  weekly player stats fetch failed, continuing without: {e}")
            weekly_stats = pd.DataFrame()

        try:
            draft_picks = nfl.load_draft_picks(seasons).to_pandas()
            logger.info(f"  draft picks: {len(draft_picks)} rows")
        except Exception as e:
            logger.warning(f"  draft picks fetch failed, continuing without: {e}")
            draft_picks = pd.DataFrame()

        return {
            'schedule': schedule,
            'pbp': pbp,
            'injuries': injuries,
            'rosters': rosters,
            'weekly_stats': weekly_stats,
            'draft_picks': draft_picks,
        }

    def _fetch_weekly_stats_per_season(self, seasons):
        frames = []
        for season in seasons:
            try:
                season_df = nfl.load_player_stats([season], summary_level="week").to_pandas()
                season_df = _downcast_floats(season_df)
                frames.append(season_df)
            except Exception as e:
                logger.warning(f"    weekly stats for {season} failed, skipping: {e}")

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def build_games_df(self, schedule_df):
        games = schedule_df[schedule_df['home_score'].notna()].copy()
        games = games.rename(columns={'game_id': 'game_id'})
        return games[['game_id', 'season', 'week', 'home_team', 'away_team',
                       'home_score', 'away_score']]

    def build_schedule_unrolled(self, schedule_df):
        played = schedule_df[schedule_df['home_score'].notna()].copy()

        home_rows = played[['season', 'week', 'home_team', 'away_team']].rename(
            columns={'home_team': 'team', 'away_team': 'opponent'}
        )
        away_rows = played[['season', 'week', 'home_team', 'away_team']].rename(
            columns={'away_team': 'team', 'home_team': 'opponent'}
        )

        return pd.concat([home_rows, away_rows], ignore_index=True)

    def prepare_weekly_stats(self, weekly_stats_df):
        if weekly_stats_df.empty:
            return weekly_stats_df

        wanted = ['player_id', 'player_name', 'position', 'team',
                  'opponent_team', 'season', 'week', 'carries', 'rushing_yards',
                  'rushing_tds', 'targets', 'receptions', 'receiving_yards',
                  'receiving_tds', 'receiving_air_yards', 'completions', 'attempts',
                  'passing_yards', 'passing_air_yards', 'passing_tds',
                  'passing_interceptions', 'sacks_suffered',
                  'fantasy_points', 'fantasy_points_ppr']

        available = [c for c in wanted if c in weekly_stats_df.columns]
        missing = set(wanted) - set(available)
        if missing:
            logger.warning(f"  weekly stats missing expected columns: {missing}")

        result = weekly_stats_df[available].rename(columns={
            'receiving_air_yards': 'air_yards',
            'team': 'recent_team',
            'passing_interceptions': 'interceptions',
            'sacks_suffered': 'sacks',
        })

        before_count = len(result)
        result = result.dropna(subset=['player_id'])
        dropped = before_count - len(result)
        if dropped > 0:
            logger.info(f"  weekly stats: dropped {dropped} rows with no player_id "
                         f"(team-level/aggregate entries, not usable for player projections)")

        return result

    def prepare_draft_picks(self, draft_picks_df):
        if draft_picks_df.empty:
            return draft_picks_df

        wanted = ['season', 'round', 'pick', 'team', 'gsis_id',
                  'pfr_player_name', 'position']
        available = [c for c in wanted if c in draft_picks_df.columns]
        missing = set(wanted) - set(available)
        if missing:
            logger.warning(f"  draft picks missing expected columns: {missing}")

        result = draft_picks_df[available].rename(columns={
            'gsis_id': 'player_id',
            'pfr_player_name': 'player_name',
        })
        return result

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

    def ensure_tables(self):
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

        CREATE TABLE IF NOT EXISTS nfl_player_weekly_stats (
            player_id TEXT,
            player_name TEXT,
            position TEXT,
            recent_team TEXT,
            opponent_team TEXT,
            season INTEGER,
            week INTEGER,
            carries DOUBLE PRECISION,
            rushing_yards DOUBLE PRECISION,
            rushing_tds DOUBLE PRECISION,
            targets DOUBLE PRECISION,
            receptions DOUBLE PRECISION,
            receiving_yards DOUBLE PRECISION,
            receiving_tds DOUBLE PRECISION,
            air_yards DOUBLE PRECISION,
            completions DOUBLE PRECISION,
            attempts DOUBLE PRECISION,
            passing_yards DOUBLE PRECISION,
            passing_air_yards DOUBLE PRECISION,
            passing_tds DOUBLE PRECISION,
            interceptions DOUBLE PRECISION,
            sacks DOUBLE PRECISION,
            fantasy_points DOUBLE PRECISION,
            fantasy_points_ppr DOUBLE PRECISION,
            updated_at TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (player_id, season, week)
        );

        ALTER TABLE nfl_player_weekly_stats ADD COLUMN IF NOT EXISTS completions DOUBLE PRECISION;
        ALTER TABLE nfl_player_weekly_stats ADD COLUMN IF NOT EXISTS attempts DOUBLE PRECISION;
        ALTER TABLE nfl_player_weekly_stats ADD COLUMN IF NOT EXISTS passing_air_yards DOUBLE PRECISION;

        CREATE TABLE IF NOT EXISTS nfl_draft_picks (
            season INTEGER,
            round INTEGER,
            pick INTEGER,
            player_id TEXT,
            player_name TEXT,
            position TEXT,
            team TEXT,
            updated_at TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (season, pick)
        );

        CREATE TABLE IF NOT EXISTS nfl_game_predictions (
            game_id TEXT PRIMARY KEY,
            season INTEGER,
            week INTEGER,
            home_team TEXT,
            away_team TEXT,
            home_rating DOUBLE PRECISION,
            away_rating DOUBLE PRECISION,
            home_win_prob DOUBLE PRECISION,
            away_win_prob DOUBLE PRECISION,
            generated_at TIMESTAMP DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS nfl_game_predictions_backtest (
            game_id TEXT,
            season INTEGER,
            week INTEGER,
            home_team TEXT,
            away_team TEXT,
            home_rating DOUBLE PRECISION,
            away_rating DOUBLE PRECISION,
            home_win_prob DOUBLE PRECISION,
            away_win_prob DOUBLE PRECISION,
            predicted_winner TEXT,
            actual_winner TEXT,
            correct BOOLEAN,
            run_label TEXT,
            generated_at TIMESTAMP DEFAULT NOW(),
        PRIMARY KEY (game_id, run_label)
        );
        """
        with self.engine.begin() as conn:
            for statement in ddl.strip().split(';'):
                if statement.strip():
                    conn.execute(text(statement))
        logger.info("Tables verified/created.")

    def upsert_by_keys(self, df, table_name, key_cols, batch_size=1000):
        if df.empty:
            logger.info(f"  {table_name}: no rows to write, skipping.")
            return

        df = df.replace({np.nan: None})

        with self.engine.begin() as conn:
            key_tuples = df[key_cols].drop_duplicates().reset_index(drop=True)
            key_cols_sql = ", ".join(f'"{k}"' for k in key_cols)

            for start in range(0, len(key_tuples), batch_size):
                batch = key_tuples.iloc[start:start + batch_size]
                row_placeholders = []
                params = {}
                for i, row in batch.iterrows():
                    placeholders = ", ".join(f":k{i}_{j}" for j in range(len(key_cols)))
                    row_placeholders.append(f"({placeholders})")
                    for j, k in enumerate(key_cols):
                        value = row[k]
                        if hasattr(value, 'item'):
                            value = value.item()
                        params[f"k{i}_{j}"] = value
                values_clause = ", ".join(row_placeholders)
                conn.execute(
                    text(f'DELETE FROM {table_name} WHERE ({key_cols_sql}) IN ({values_clause})'),
                    params
                )

            df.to_sql(table_name, conn, if_exists='append', index=False)

        logger.info(f"  {table_name}: wrote {len(df)} rows.")

    def sync_to_s3(self, df, key):
        if not self.s3_client:
            return
        buffer = BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        self.s3_client.put_object(Bucket=self.s3_bucket, Key=key, Body=buffer.getvalue())
        logger.info(f"  synced to s3://{self.s3_bucket}/{key}")

    def run(self, seasons, run_label):
        source = self.fetch_source_data(seasons)

        games_df = self.build_games_df(source['schedule'])
        schedule_unrolled = self.build_schedule_unrolled(source['schedule'])

        elo_history, epa_features, unit_ratings = self.run_feature_engineering(
            games_df, source['pbp'], schedule_unrolled
        )

        weekly_stats = self.prepare_weekly_stats(source['weekly_stats'])
        draft_picks = self.prepare_draft_picks(source['draft_picks'])

        self.ensure_tables()

        logger.info("Writing to Postgres...")
        self.upsert_by_keys(elo_history, 'nfl_elo_ratings', ['game_id'])
        self.upsert_by_keys(epa_features, 'nfl_epa_features', ['team', 'season', 'week'])
        self.upsert_by_keys(unit_ratings, 'nfl_unit_ratings', ['team', 'season', 'week'])
        self.upsert_by_keys(weekly_stats, 'nfl_player_weekly_stats',
                             ['player_id', 'season', 'week'])
        self.upsert_by_keys(draft_picks, 'nfl_draft_picks', ['season', 'pick'])

        if self.s3_client:
            logger.info("Syncing snapshots to S3...")
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            self.sync_to_s3(elo_history, f'nfl/{run_label}/elo_{timestamp}.parquet')
            self.sync_to_s3(epa_features, f'nfl/{run_label}/epa_{timestamp}.parquet')
            self.sync_to_s3(unit_ratings, f'nfl/{run_label}/unit_ratings_{timestamp}.parquet')

        logger.info(f"Pipeline run complete ({run_label}).")

    def generate_upcoming_predictions(self, season):
        """
        Runs after the main weekly update — uses whatever team ratings
        resulted from processing this week's completed games to predict
        the NEXT upcoming week. Re-derives current ratings from the full
        completed-games history on a fresh engine instance (simplest
        correct approach; not the most efficient, worth revisiting later).
        """
        schedule = nfl.load_schedules([season]).to_pandas()

        upcoming = schedule[schedule['home_score'].isna()].copy()
        if upcoming.empty:
            logger.info("  No upcoming games found — season may be complete.")
            return

        next_week = upcoming['week'].min()
        next_week_games = upcoming[upcoming['week'] == next_week]

        elo_engine = NFLEloEngine()
        completed_games = self.build_games_df(schedule)
        elo_engine.process_games(completed_games)

        predictions = elo_engine.predict_upcoming_games(next_week_games)
        logger.info(f"  Generated predictions for {len(predictions)} games (week {next_week})")

        self.upsert_by_keys(predictions, 'nfl_game_predictions', ['game_id'])

    def run_weekly(self):
        season = determine_current_season()
        logger.info(f"=== WEEKLY RUN — season {season} ===")
        self.run(seasons=[season], run_label='weekly')

        logger.info("Generating predictions for upcoming games...")
        self.generate_upcoming_predictions(season)

    def run_backfill(self, start_year, end_year):
        seasons = list(range(start_year, end_year + 1))
        if len(seasons) > 2:
            logger.info(
                f"Backfill spans {len(seasons)} seasons ({seasons[0]}-{seasons[-1]}). "
                "Consider testing a smaller range first."
            )
        logger.info(f"=== BACKFILL RUN — seasons {seasons} ===")
        self.run(seasons=seasons, run_label='backfill')


def main():
    parser = argparse.ArgumentParser(description="Cortex Sports Analytics — NFL Pipeline")
    parser.add_argument('--mode', choices=['weekly', 'backfill'], required=True)
    parser.add_argument('--start-year', type=int, default=None)
    parser.add_argument('--end-year', type=int, default=None)
    args = parser.parse_args()

    current_season = determine_current_season()
    run_label = args.mode
    seasons_ctx = None

    try:
        pipeline = NFLPipeline()

        if args.mode == 'weekly':
            seasons_ctx = [current_season]
            pipeline.run_weekly()
        else:
            end_year = args.end_year if args.end_year is not None else current_season - 1
            start_year = args.start_year if args.start_year is not None else current_season - 5
            seasons_ctx = list(range(start_year, end_year + 1))
            pipeline.run_backfill(start_year, end_year)

    except Exception as e:
        logger.exception(f"Pipeline run failed ({run_label}).")
        send_failure_alert(run_label, e, seasons_ctx)
        sys.exit(1)


if __name__ == "__main__":
    main()
