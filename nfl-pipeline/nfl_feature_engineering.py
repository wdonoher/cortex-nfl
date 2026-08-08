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

EXPECTED INPUT SCHEMAS
-----------------------
games_df (one row per game, final results):
    game_id, season, week, game_date
    home_team, away_team, home_score, away_score

pbp_df (play-by-play, NFLFastR-style):
    game_id, season, week, posteam, defteam
    play_type ('pass','run', etc.), epa
    pass_attempt, complete_pass, sack, qb_hit, pressure (bool/flag)
    rush_attempt, yards_before_contact (if available)
    down, ydstogo, yardline_100

injury_df / usage_df: as defined in the pattern_flags module — used only
by the injury-adjustment hook, not required for baseline ratings.
"""

import pandas as pd
import numpy as np


# ============================================================================
# 1. ELO RATING ENGINE
# ============================================================================

class NFLEloEngine:
    """
    Standard Elo with NFL-specific adjustments: home field advantage and
    a margin-of-victory multiplier (blowouts move ratings more than
    narrow wins, but with diminishing returns so running up the score
    doesn't distort things).
    """

    def __init__(self, k_factor=20, home_advantage=48, initial_rating=1500,
                 season_regression=1/3, use_mov_multiplier=True):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.season_regression = season_regression
        self.use_mov_multiplier = use_mov_multiplier
        self.ratings = {}  # team -> current rating

    def get_rating(self, team):
        return self.ratings.get(team, self.initial_rating)

    def expected_score(self, rating_a, rating_b):
        """Probability team A beats team B given the two ratings."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def _mov_multiplier(self, point_diff, elo_diff):
        """
        Margin-of-victory multiplier, adapted from the FiveThirtyEight
        NFL Elo formula. Dampens blowout impact against already-huge
        favorites, amplifies upset margins.
        """
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
        """Call between seasons — pulls all ratings partway back to 1500."""
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
        for every game — this is what feeds the model training pipeline.
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


# ============================================================================
# 2. EPA FEATURE ENGINE
# ============================================================================

class EPAFeatureEngine:
    """
    Computes rolling offensive/defensive EPA per play, with trend slope,
    from play-by-play data. This is team-level (Tier 0 / baseline), and
    also serves as the raw input several Tier 1 unit metrics are built from.
    """

    def __init__(self, window=4):
        self.window = window

    def _team_week_epa(self, pbp_df):
        """Aggregate play-by-play into team-week offensive/defensive EPA/play."""
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
        """
        Returns team-week EPA features: rolling offensive/defensive EPA
        per play, plus trend slope for each. This is the primary input
        to both the win-probability/margin model and the Tier 1 unit
        rating calculations below.
        """
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

    Methodology (MVP — see framework doc for the full rationale):
      1. Compute each team's raw per-unit metric for the week.
      2. Opponent-adjust: subtract the opponent's own season-to-date
         average allowed/generated value for that metric, so a good
         week against a tough opponent counts for more than the same
         raw number against a weak one.
      3. Roll the opponent-adjusted values over a trailing window.
      4. Z-score across the league for that week, so units are
         comparable on a common scale (roughly -3 to +3, 0 = league avg).
      5. Trend slope, same as the EPA engine.

    This is intentionally simpler than a full iterative Elo-style system
    (that's the planned v2 upgrade) but captures opponent strength and
    recent form, which raw averages don't.
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
        """
        Derives the raw per-team, per-week metrics feeding each unit.
        Expects pbp_df to have: posteam, defteam, season, week, epa,
        play_type, pass_attempt, sack, qb_hit/pressure flag, rush_attempt.
        """
        pbp = pbp_df.copy()

        # Ensure a pressure flag exists (sack or qb_hit as proxy if a
        # dedicated 'pressure' column isn't present)
        if 'pressure' not in pbp.columns:
            pbp['pressure'] = (
                (pbp.get('sack', 0) == 1) | (pbp.get('qb_hit', 0) == 1)
            ).astype(int)

        pass_plays = pbp[pbp['pass_attempt'] == 1]
        rush_plays = pbp[pbp['rush_attempt'] == 1]

        # Offense: pass protection (pressure allowed) and passing efficiency
        off_pass = pass_plays.groupby(['posteam', 'season', 'week']).agg(
            pressure_rate_allowed=('pressure', 'mean'),
            epa_per_dropback=('epa', 'mean')
        ).reset_index().rename(columns={'posteam': 'team'})

        # Defense: pass rush (pressure generated) and pass defense efficiency
        def_pass = pass_plays.groupby(['defteam', 'season', 'week']).agg(
            pressure_rate_generated=('pressure', 'mean'),
            epa_per_dropback_allowed=('epa', 'mean')
        ).reset_index().rename(columns={'defteam': 'team'})

        # Offense: run blocking efficiency
        off_rush = rush_plays.groupby(['posteam', 'season', 'week']).agg(
            epa_per_rush=('epa', 'mean')
        ).reset_index().rename(columns={'posteam': 'team'})

        # Defense: run defense efficiency
        def_rush = rush_plays.groupby(['defteam', 'season', 'week']).agg(
            epa_per_rush_allowed=('epa', 'mean')
        ).reset_index().rename(columns={'defteam': 'team'})

        raw = off_pass.merge(def_pass, on=['team', 'season', 'week'], how='outer')
        raw = raw.merge(off_rush, on=['team', 'season', 'week'], how='outer')
        raw = raw.merge(def_rush, on=['team', 'season', 'week'], how='outer')

        return raw.sort_values(['team', 'season', 'week'])

    def _opponent_adjust(self, raw_df, metric_col, schedule_df):
        """
        Subtracts the opponent's season-to-date average for the
        complementary side of the metric, so performance is measured
        relative to who was faced. Requires schedule_df to map each
        team-week to its opponent.
        """
        df = raw_df.merge(
            schedule_df[['team', 'season', 'week', 'opponent']],
            on=['team', 'season', 'week'], how='left'
        )

        # Opponent's season-to-date average for this metric (entering the week)
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

        # Z-score across the league within each week
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
        """
        schedule_df expected columns: team, season, week, opponent
        (one row per team per game — i.e. the schedule "unrolled" so each
        team has its own row with its opponent that week).

        Returns team-week unit ratings: opponent-adjusted, rolled,
        z-scored, with trend slope — for each of the six Tier 1 units.
        """
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
        """
        HOOK for injury-adjusted ratings (baseline vs this-week-adjusted),
        per the framework doc. Not yet wired to live data.

        unit_position_map: dict mapping unit_name -> list of positions
            e.g. {'pass_protection': ['LT','LG','C','RG','RT'], ...}
        usage_df: player-week usage (snap_pct or target_share/carry_share)
            used to weight injury impact by the player's share of the unit.

        Returns unit_ratings_df with added `{unit}_rating_injury_adj` columns.
        Left as a stub — implement once injury_df / usage_df are flowing
        from the live pipeline with confirmed column names.
        """
        raise NotImplementedError(
            "Wire this up once injury_df and usage_df schemas are confirmed "
            "from the live data pipeline."
        )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # games_df = load_games(seasons=[2020, 2021, 2022, 2023, 2024, 2025])
    # elo_engine = NFLEloEngine()
    # elo_history = elo_engine.process_games(games_df)

    # pbp_df = load_pbp(seasons=[2023, 2024, 2025])
    # epa_engine = EPAFeatureEngine(window=4)
    # epa_features = epa_engine.compute(pbp_df)

    # schedule_unrolled = load_schedule_unrolled(seasons=[2023, 2024, 2025])
    # unit_engine = UnitRatingEngine(window=4)
    # unit_ratings = unit_engine.compute_unit_ratings(pbp_df, schedule_unrolled)

    print("Elo, EPA, and Tier 1 Unit Rating engines ready.")
    print("Wire in historical games_df / pbp_df / schedule_df to generate features.")
