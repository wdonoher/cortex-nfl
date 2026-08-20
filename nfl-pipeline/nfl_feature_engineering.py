"""
Cortex Sports Analytics — NFL Feature Engineering
==================================================
Implements the three foundational rating layers agreed in the matchup
decomposition framework:

  1. NFLEloEngine        — team-level power rating (game-level, iterative)
  2. EPAFeatureEngine     — rolling EPA efficiency (play-by-play derived)
  3. UnitRatingEngine     — Tier 1 unit-vs-unit ratings (opponent-adjusted
                            rolling z-scores, with trend slope + injury hook)

No odds/Vegas inputs anywhere in this module — every rating is derived
from game results and play-by-play data.
"""

import pandas as pd
import numpy as np


# ============================================================================
# 1. ELO RATING ENGINE
# ============================================================================

class NFLEloEngine:
    """
    Standard Elo with NFL-specific adjustments: home field advantage and
    a margin-of-victory multiplier.
    """

    def __init__(self, k_factor=20, home_advantage=48, initial_rating=1500,
                 season_regression=1/3, use_mov_multiplier=True):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.season_regression = season_regression
        self.use_mov_multiplier = use_mov_multiplier
        self.ratings = {}

    def get_rating(self, team):
        return self.ratings.get(team, self.initial_rating)

    def expected_score(self, rating_a, rating_b):
        """Probability team A beats team B given the two ratings."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def _mov_multiplier(self, point_diff, elo_diff):
        if not self.use_mov_multiplier:
            return 1.0
        return np.log(max(abs(point_diff), 1) + 1) * (
            2.2 / (0.001 * elo_diff + 2.2)
        )

    def update_single_game(self, home_team, away_team, home_score, away_score):
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)

        home_rating_adj = home_rating + self.home_advantage
        expected_home = self.expected_score(home_rating_adj, away_rating)

        if home_score > away_score:
            actual_home = 1.0
        elif home_score < away_score:
            actual_home = 0.0
        else:
            actual_home = 0.5

        point_diff = home_score - away_score
        elo_diff = home_rating_adj - away_rating
        mult = self._mov_multiplier(point_diff, elo_diff)

        shift = self.k_factor * mult * (actual_home - expected_home)

        new_home = home_rating + shift
        new_away = away_rating - shift

        self.ratings[home_team] = new_home
        self.ratings[away_team] = new_away

        return {
            'home_rating_pre': home_rating,
            'away_rating_pre': away_rating,
            'home_rating_post': new_home,
            'away_rating_post': new_away,
            'home_win_prob_pre': expected_home,
        }

    def regress_to_mean(self, factor=None):
        factor = factor if factor is not None else self.season_regression
        for team in self.ratings:
            self.ratings[team] = (
                self.ratings[team] * (1 - factor) + self.initial_rating * factor
            )

    def process_games(self, games_df):
        """
        Processes a season (or multiple seasons) of games chronologically,
        applying inter-season regression when the season number changes.
        Returns a DataFrame with pre/post ratings and pre-game win prob
        for every game.
        """
        games_df = games_df.sort_values(['season', 'week']).reset_index(drop=True)
        records = []
        current_season = None

        for _, g in games_df.iterrows():
            if current_season is not None and g['season'] != current_season:
                self.regress_to_mean()
            current_season = g['season']

            result = self.update_single_game(
                g['home_team'], g['away_team'], g['home_score'], g['away_score']
            )
            records.append({
                'game_id': g['game_id'],
                'season': g['season'],
                'week': g['week'],
                'home_team': g['home_team'],
                'away_team': g['away_team'],
                **result
            })

        return pd.DataFrame(records)

    def predict_upcoming_games(self, upcoming_games_df):
        """
        Forward-looking win probability for games that haven't been
        played yet, using CURRENT team ratings (self.ratings, as of
        whatever games have already been processed). Unlike
        update_single_game(), this is READ-ONLY — it doesn't update
        ratings, since there's no result yet to learn from.

        upcoming_games_df expected columns: game_id, season, week,
        home_team, away_team (no scores — these haven't been played).
        """
        predictions = []
        for _, g in upcoming_games_df.iterrows():
            home_rating = self.get_rating(g['home_team'])
            away_rating = self.get_rating(g['away_team'])
            home_rating_adj = home_rating + self.home_advantage
            home_win_prob = self.expected_score(home_rating_adj, away_rating)

            predictions.append({
                'game_id': g['game_id'],
                'season': g['season'],
                'week': g['week'],
                'home_team': g['home_team'],
                'away_team': g['away_team'],
                'home_rating': home_rating,
                'away_rating': away_rating,
                'home_win_prob': home_win_prob,
                'away_win_prob': 1 - home_win_prob,
            })

        return pd.DataFrame(predictions)


# ============================================================================
# 2. EPA FEATURE ENGINE
# ============================================================================

class EPAFeatureEngine:
    """
    Computes rolling offensive/defensive EPA per play, with trend slope,
    from play-by-play data.
    """

    def __init__(self, window=4):
        self.window = window

    def _team_week_epa(self, pbp_df):
        off = pbp_df.groupby(['posteam', 'season', 'week'])['epa'].mean().reset_index()
        off = off.rename(columns={'posteam': 'team', 'epa': 'epa_off_play'})

        deff = pbp_df.groupby(['defteam', 'season', 'week'])['epa'].mean().reset_index()
        deff = deff.rename(columns={'defteam': 'team', 'epa': 'epa_def_play_allowed'})

        merged = off.merge(deff, on=['team', 'season', 'week'], how='outer')
        return merged.sort_values(['team', 'season', 'week'])

    def _rolling_and_trend(self, df, value_col, group_col='team'):
        df = df.sort_values([group_col, 'season', 'week']).copy()

        df[f'{value_col}_roll{self.window}'] = (
            df.groupby(group_col)[value_col]
            .transform(lambda x: x.rolling(self.window, min_periods=1).mean())
        )

        def slope(x):
            if len(x) < 2:
                return 0.0
            idx = np.arange(len(x))
            return np.polyfit(idx, x, 1)[0]

        df[f'{value_col}_trend_slope'] = (
            df.groupby(group_col)[value_col]
            .transform(lambda x: x.rolling(self.window, min_periods=2)
                       .apply(slope, raw=True))
        )

        return df

    def compute(self, pbp_df):
        team_week = self._team_week_epa(pbp_df)
        team_week = self._rolling_and_trend(team_week, 'epa_off_play')
        team_week = self._rolling_and_trend(team_week, 'epa_def_play_allowed')
        return team_week


# ============================================================================
# 3. TIER 1 UNIT RATING ENGINE
# ============================================================================

class UnitRatingEngine:
    """
    Opponent-adjusted rolling z-score ratings for the Tier 1 units:
    pass protection, pass rush, passing offense, pass defense,
    run blocking, run defense.
    """

    UNIT_METRIC_MAP = {
        'pass_protection': {'source': 'offense', 'metric': 'pressure_rate_allowed'},
        'pass_rush':       {'source': 'defense', 'metric': 'pressure_rate_generated'},
        'passing_offense': {'source': 'offense', 'metric': 'epa_per_dropback'},
        'pass_defense':    {'source': 'defense', 'metric': 'epa_per_dropback_allowed'},
        'run_blocking':    {'source': 'offense', 'metric': 'epa_per_rush'},
        'run_defense':     {'source': 'defense', 'metric': 'epa_per_rush_allowed'},
    }

    def __init__(self, window=4):
        self.window = window

    def compute_raw_unit_metrics(self, pbp_df):
        pbp = pbp_df.copy()

        if 'pressure' not in pbp.columns:
            pbp['pressure'] = (
                (pbp.get('sack', 0) == 1) | (pbp.get('qb_hit', 0) == 1)
            ).astype(int)

        pass_plays = pbp[pbp['pass_attempt'] == 1]
        rush_plays = pbp[pbp['rush_attempt'] == 1]

        off_pass = pass_plays.groupby(['posteam', 'season', 'week']).agg(
            pressure_rate_allowed=('pressure', 'mean'),
            epa_per_dropback=('epa', 'mean')
        ).reset_index().rename(columns={'posteam': 'team'})

        def_pass = pass_plays.groupby(['defteam', 'season', 'week']).agg(
            pressure_rate_generated=('pressure', 'mean'),
            epa_per_dropback_allowed=('epa', 'mean')
        ).reset_index().rename(columns={'defteam': 'team'})

        off_rush = rush_plays.groupby(['posteam', 'season', 'week']).agg(
            epa_per_rush=('epa', 'mean')
        ).reset_index().rename(columns={'posteam': 'team'})

        def_rush = rush_plays.groupby(['defteam', 'season', 'week']).agg(
            epa_per_rush_allowed=('epa', 'mean')
        ).reset_index().rename(columns={'defteam': 'team'})

        raw = off_pass.merge(def_pass, on=['team', 'season', 'week'], how='outer')
        raw = raw.merge(off_rush, on=['team', 'season', 'week'], how='outer')
        raw = raw.merge(def_rush, on=['team', 'season', 'week'], how='outer')

        return raw.sort_values(['team', 'season', 'week'])

    def _opponent_adjust(self, raw_df, metric_col, schedule_df):
        df = raw_df.merge(
            schedule_df[['team', 'season', 'week', 'opponent']],
            on=['team', 'season', 'week'], how='left'
        )

        opp_avg = raw_df[['team', 'season', 'week', metric_col]].rename(
            columns={'team': 'opponent', metric_col: 'opponent_avg_metric'}
        )
        opp_avg = opp_avg.sort_values(['opponent', 'season', 'week'])
        opp_avg['opponent_szn_avg_entering'] = (
            opp_avg.groupby(['opponent', 'season'])['opponent_avg_metric']
            .transform(lambda x: x.expanding().mean().shift(1))
        )

        df = df.merge(
            opp_avg[['opponent', 'season', 'week', 'opponent_szn_avg_entering']],
            on=['opponent', 'season', 'week'], how='left'
        )

        league_avg = raw_df[metric_col].mean()
        df['opponent_szn_avg_entering'] = df['opponent_szn_avg_entering'].fillna(league_avg)

        df[f'{metric_col}_adj'] = (
            df[metric_col] - df['opponent_szn_avg_entering'] + league_avg
        )

        return df

    def _rolling_zscore_and_trend(self, df, adj_col):
        df = df.sort_values(['team', 'season', 'week']).copy()

        df[f'{adj_col}_roll'] = (
            df.groupby('team')[adj_col]
            .transform(lambda x: x.rolling(self.window, min_periods=1).mean())
        )

        df[f'{adj_col}_zscore'] = (
            df.groupby(['season', 'week'])[f'{adj_col}_roll']
            .transform(lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else 0)
        )

        def slope(x):
            if len(x) < 2:
                return 0.0
            idx = np.arange(len(x))
            return np.polyfit(idx, x, 1)[0]

        df[f'{adj_col}_trend_slope'] = (
            df.groupby('team')[f'{adj_col}_roll']
            .transform(lambda x: x.rolling(self.window, min_periods=2)
                       .apply(slope, raw=True))
        )

        return df

    def compute_unit_ratings(self, pbp_df, schedule_df):
        raw = self.compute_raw_unit_metrics(pbp_df)

        result = raw[['team', 'season', 'week']].drop_duplicates()

        for unit_name, cfg in self.UNIT_METRIC_MAP.items():
            metric_col = cfg['metric']
            adjusted = self._opponent_adjust(raw, metric_col, schedule_df)
            adjusted = self._rolling_zscore_and_trend(adjusted, f'{metric_col}_adj')

            keep_cols = ['team', 'season', 'week',
                         f'{metric_col}_adj_zscore', f'{metric_col}_adj_trend_slope']
            renamed = adjusted[keep_cols].rename(columns={
                f'{metric_col}_adj_zscore': f'{unit_name}_rating',
                f'{metric_col}_adj_trend_slope': f'{unit_name}_trend'
            })

            result = result.merge(renamed, on=['team', 'season', 'week'], how='left')

        return result

    def apply_injury_adjustment(self, unit_ratings_df, injury_df, usage_df,
                                 unit_position_map):
        raise NotImplementedError(
            "Wire this up once injury_df and usage_df schemas are confirmed "
            "from the live data pipeline."
        )


if __name__ == "__main__":
    print("Elo, EPA, and Tier 1 Unit Rating engines ready.")
    print("Wire in historical games_df / pbp_df / schedule_df to generate features.")
