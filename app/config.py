"""Configuration constants for the T20 Bowling Strategy Analyzer"""

# Page names
PAGE_BATTER_ANALYSIS = "Batter Analysis"
PAGE_BOWLER_STRATEGIES = "Bowler Strategies"
PAGE_MATCHUP_OPTIMIZATION = "Match-up Optimization"
PAGE_BOWLING_PLAN = "Complete Bowling Plan"
PAGE_SMART_STATS = "Smart Stats"

# Navigation options
DEFAULT_PAGE = PAGE_BATTER_ANALYSIS

# Game phases
PHASE_NAMES = {
    1: "Powerplay (Overs 1-6)",
    2: "Early Middle (Overs 7-12)",
    3: "Late Middle (Overs 13-16)", 
    4: "Death (Overs 17-20)"
}

# Visualization settings
PLOT_FIGSIZE = (10, 6)
HEATMAP_FIGSIZE = (12, 8)
FIELD_FIGSIZE = (8, 8)

# Analysis thresholds
MIN_BALLS_FOR_ANALYSIS = 5  # Minimum balls faced/bowled for meaningful analysis
MIN_DISMISSALS_FOR_AVERAGE = 1  # Minimum dismissals needed to calculate average

# UI Constants
PHASE_EMPHASIS_OPTIONS = ["Powerplay", "Balanced", "Death Overs"]
STRATEGY_EMPHASIS_OPTIONS = ["Economy", "Balanced", "Wicket-taking"]