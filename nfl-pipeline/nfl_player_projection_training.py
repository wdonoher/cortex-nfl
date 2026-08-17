"""
Cortex Sports Analytics — NFL Player Fantasy Points Projection Model
=====================================================================
Per-position XGBoost regressors (QB/RB/WR/TE) producing the headline
projected-fantasy-points number the sit/start comparison tool depends
on. Team/unit ratings (Elo, EPA, Tier 1) feed in as matchup-adjustment
features — they're inputs here, not a substitute for this output.

TARGET: fantasy_points_ppr, as returned directly by nfl_data_py's
import_weekly_data() (already flowing into nfl_player_weekly_stats
via the main pipeline).

ARCHITECTURE NOTE: This is a TRAINING script, meant to run periodically
(e.g. monthly, or once preseason + a few times through the year) as a
one-off Railway job (same start-command-swap + Run Now pattern used
for the historical backfill) — NOT part of the weekly cron pipeline.
The weekly job loads the saved model artifacts and calls
generate_weekly_projections(); it does not retrain.

DATA SOURCE: reads directly from Postgres (nfl_player_weekly_stats,
nfl_unit_ratings) — both populated by nfl_pipeline.py's backfill/weekly
runs. No separate data fetching happens in this file.
"""

import argparse
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import io
import os
from sqlalchemy import create_engine, text

TARGET_COL = 'fantasy_points_ppr'

POSITION_GROUPS = ['QB', 'RB', 'WR', 'TE']

POSITION_MATCHUP_UNITS = {
    'QB': ['pass_defense_rating', 'pass_defense_trend',
           'pass_rush_rating', 'pass_rush_trend'],
    'RB': ['run_defense_rating', 'run_defense_trend',
           'pass_defense_rating', 'pass_defense_trend'],
    'WR': ['pass_defense_rating', 'pass_defense_trend'],
    'TE': ['pass_defense_rating', 'pass_defense_trend'],
}

ROLLING_STAT_COLS = {
    'QB': ['passing_yards', 'passing_tds', 'interceptions', 'sacks', TARGET_COL],
    'RB': ['carries', 'rushing_yards', 'rushing_tds',
           'targets', 'receptions', 'receiving_yards', TARGET_COL],
    'WR': ['targets', 'receptions', 'receiving_yards', 'receiving_tds',
           'air_yards', TARGET_COL],
    'TE': ['targets', 'receptions', 'receiving_yards', 'receiving_tds', TARGET_COL],
}


# ============================================================================
# DATABASE CONNECTION / MODEL STORAGE (Postgres)
# ============================================================================

def get_engine():
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL not set. Required to read training data and "
            "read/write trained models from Postgres."
        )
    return create_engine(database_url)


def ensure_model_table(engine):
    ddl = """
    CREATE TABLE IF NOT EXISTS nfl_trained_models (
        position TEXT PRIMARY KEY,
        trained_at TIMESTAMP DEFAULT NOW(),
        model_bytes BYTEA,
        feature_cols TEXT[]
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def load_training_data_from_db(engine):
    """
    Pulls the two tables training needs directly from Postgres — both
    already populated by nfl_pipeline.py's backfill run.
    """
    weekly_df = pd.read_sql("SELECT * FROM nfl_player_weekly_stats", engine)
    unit_ratings_df = pd.read_sql("SELECT * FROM nfl_unit_ratings", engine)
    return weekly_df, unit_ratings_df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def build_rolling_player_features(weekly_df, position, window=4):
    """
    Rolling averages of the position's relevant raw stats, computed
    using only PRIOR weeks (shifted) so the model never sees the
    current week's own outcome as an input feature.
    """
    pos_df = weekly_df[weekly_df['position'] == position].copy()
    pos_df = pos_df.sort_values(['player_id', 'season', 'week'])

    stat_cols = [c for c in ROLLING_STAT_COLS[position] if c in pos_df.columns]

    for col in stat_cols:
        pos_df[f'{col}_roll{window}'] = (
            pos_df.groupby('player_id')[col]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    return pos_df


def attach_matchup_features(pos_df, unit_ratings_df, position):
    """
    Joins the opponent's relevant Tier 1 unit ratings for that week —
    this is what makes the projection matchup-adjusted rather than
    just a rolling average of the player's own history.
    """
    matchup_cols = POSITION_MATCHUP_UNITS[position]
    opp_ratings = unit_ratings_df[['team', 'season', 'week'] + matchup_cols].rename(
        columns={'team': 'opponent_team'}
    )

    merged = pos_df.merge(
        opp_ratings, on=['opponent_team', 'season', 'week'], how='left'
    )

    for col in matchup_cols:
        merged[col] = merged[col].fillna(0)

    return merged


def build_training_dataset(weekly_df, unit_ratings_df, position, window=4,
                            min_snap_threshold=None):
    """
    Full pipeline: rolling player features + matchup features + target.
    Drops the first appearance of each player (no rolling history yet —
    this is the known rookie cold-start gap, handled separately by the
    rookie-handling Phase 1 cohort model, not this file) and any row
    missing the target.
    """
    pos_df = build_rolling_player_features(weekly_df, position, window)
    pos_df = attach_matchup_features(pos_df, unit_ratings_df, position)

    roll_cols = [f'{c}_roll{window}' for c in ROLLING_STAT_COLS[position]
                 if c in weekly_df.columns]
    matchup_cols = POSITION_MATCHUP_UNITS[position]

    feature_cols = roll_cols + matchup_cols

    pos_df = pos_df.dropna(subset=[TARGET_COL])
    pos_df = pos_df.dropna(subset=roll_cols, how='all')

    if min_snap_threshold and 'snap_pct' in pos_df.columns:
        pos_df = pos_df[pos_df['snap_pct'] >= min_snap_threshold]

    return pos_df, feature_cols


# ============================================================================
# MODEL
# ============================================================================

class PositionProjectionModel:
    """Trains, evaluates, and serves projections for one position group."""

    def __init__(self, position, model_params=None):
        self.position = position
        default_params = {
            'n_estimators': 300,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
        }
        self.model_params = {**default_params, **(model_params or {})}
        self.model = XGBRegressor(**self.model_params)
        self.feature_cols = None

    def train(self, train_df, feature_cols):
        self.feature_cols = feature_cols
        X = train_df[feature_cols]
        y = train_df[TARGET_COL]
        self.model.fit(X, y)

    def predict(self, df):
        X = df[self.feature_cols]
        preds = self.model.predict(X)
        return np.maximum(preds, 0)

    def evaluate(self, test_df):
        preds = self.predict(test_df)
        actual = test_df[TARGET_COL]

        mae = mean_absolute_error(actual, preds)
        rmse = np.sqrt(mean_squared_error(actual, preds))
        r2 = r2_score(actual, preds)

        print(f"[{self.position}] MAE: {mae:.2f}  RMSE: {rmse:.2f}  R²: {r2:.3f}  "
              f"(n={len(test_df)})")

        return {'position': self.position, 'mae': mae, 'rmse': rmse, 'r2': r2,
                'n_test': len(test_df)}

    def feature_importance(self, top_n=10):
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        return importance

    def save(self, engine=None):
        """
        Serializes the model to bytes and upserts it into the
        nfl_trained_models Postgres table — no local disk involved,
        so this survives a one-off job's container being torn down.
        """
        engine = engine or get_engine()
        ensure_model_table(engine)

        buffer = io.BytesIO()
        joblib.dump({'model': self.model, 'feature_cols': self.feature_cols}, buffer)
        model_bytes = buffer.getvalue()

        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM nfl_trained_models WHERE position = :position"),
                {'position': self.position}
            )
            conn.execute(
                text(
                    "INSERT INTO nfl_trained_models (position, model_bytes, feature_cols) "
                    "VALUES (:position, :model_bytes, :feature_cols)"
                ),
                {
                    'position': self.position,
                    'model_bytes': model_bytes,
                    'feature_cols': self.feature_cols,
                }
            )

        return f"Saved {self.position} model to Postgres (nfl_trained_models)."

    @classmethod
    def load(cls, position, engine=None):
        """Loads a trained model's bytes back out of Postgres and deserializes it."""
        engine = engine or get_engine()

        with engine.begin() as conn:
            result = conn.execute(
                text(
                    "SELECT model_bytes, feature_cols FROM nfl_trained_models "
                    "WHERE position = :position"
                ),
                {'position': position}
            ).fetchone()

        if result is None:
            raise FileNotFoundError(
                f"No saved model found for position '{position}' in "
                f"nfl_trained_models. Has training been run yet?"
            )

        model_bytes, feature_cols = result
        buffer = io.BytesIO(model_bytes)
        payload = joblib.load(buffer)

        instance = cls(position)
        instance.model = payload['model']
        instance.feature_cols = payload['feature_cols']
        return instance


# ============================================================================
# TRAIN ALL POSITIONS
# ============================================================================

def train_all_positions(weekly_df, unit_ratings_df, train_seasons, test_season,
                         window=4, engine=None):
    """
    Trains one model per position group with a TIME-BASED split —
    train on train_seasons, evaluate on test_season. This mimics real
    deployment (predicting a future week from past data) rather than
    a random shuffle split.
    """
    engine = engine or get_engine()
    results = {}

    for position in POSITION_GROUPS:
        print(f"\n{'='*50}\nTraining {position} model\n{'='*50}")

        pos_df, feature_cols = build_training_dataset(
            weekly_df, unit_ratings_df, position, window
        )

        train_df = pos_df[pos_df['season'].isin(train_seasons)]
        test_df = pos_df[pos_df['season'] == test_season]

        if train_df.empty or test_df.empty:
            print(f"  Skipping {position}: insufficient data "
                  f"(train={len(train_df)}, test={len(test_df)})")
            continue

        model = PositionProjectionModel(position)
        model.train(train_df, feature_cols)
        metrics = model.evaluate(test_df)

        print(model.feature_importance())

        save_msg = model.save(engine)
        print(f"  {save_msg}")

        results[position] = {'model': model, 'metrics': metrics}

    return results


# ============================================================================
# WEEKLY INFERENCE (called from the weekly pipeline, not this script)
# ============================================================================

def generate_weekly_projections(weekly_df, unit_ratings_df, target_season,
                                 target_week, engine=None, window=4):
    """
    Produces this week's projections for every active player, using
    the SAVED models (no training here).
    """
    engine = engine or get_engine()
    all_projections = []

    for position in POSITION_GROUPS:
        try:
            model = PositionProjectionModel.load(position, engine)
        except FileNotFoundError:
            print(f"No saved model for {position} — skipping.")
            continue

        pos_df, feature_cols = build_training_dataset(
            weekly_df, unit_ratings_df, position, window
        )

        current_week = pos_df[
            (pos_df['season'] == target_season) & (pos_df['week'] == target_week)
        ].copy()

        if current_week.empty:
            print(f"No {position} rows found for {target_season} week {target_week}.")
            continue

        current_week['projected_fantasy_points'] = model.predict(current_week)

        all_projections.append(
            current_week[['player_id', 'player_name', 'position', 'recent_team',
                           'opponent_team', 'season', 'week',
                           'projected_fantasy_points']]
            .rename(columns={'recent_team': 'team'})
        )

    if not all_projections:
        return pd.DataFrame()

    return pd.concat(all_projections, ignore_index=True).sort_values(
        'projected_fantasy_points', ascending=False
    )


# ============================================================================
# CLI ENTRY POINT — this is what actually runs as the one-off training job
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cortex NFL Player Projection Model — Training"
    )
    parser.add_argument('--train-start', type=int, default=2021)
    parser.add_argument('--train-end', type=int, default=2024)
    parser.add_argument('--test-season', type=int, default=2025)
    args = parser.parse_args()

    engine = get_engine()

    print("Loading training data from Postgres...")
    weekly_df, unit_ratings_df = load_training_data_from_db(engine)
    print(f"  nfl_player_weekly_stats: {len(weekly_df)} rows")
    print(f"  nfl_unit_ratings: {len(unit_ratings_df)} rows")

    train_seasons = list(range(args.train_start, args.train_end + 1))
    print(f"\nTraining on seasons {train_seasons}, evaluating against {args.test_season}")

    results = train_all_positions(
        weekly_df, unit_ratings_df,
        train_seasons=train_seasons,
        test_season=args.test_season,
        engine=engine
    )

    print(f"\n{'='*50}\nTraining Summary\n{'='*50}")
    for position, r in results.items():
        m = r['metrics']
        print(f"  {position}: MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  "
              f"R²={m['r2']:.3f}  (n_test={m['n_test']})")

    if not results:
        print("  No models were successfully trained — check data availability.")


if __name__ == "__main__":
    main()
