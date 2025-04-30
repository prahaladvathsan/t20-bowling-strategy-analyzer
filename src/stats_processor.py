"""Base stats processor for cricket analytics"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from functools import lru_cache

@dataclass
class BattingStats:
    """Container for batting statistics"""
    runs: int
    balls: int
    dismissals: int
    dot_balls: int
    singles: int
    fours: int
    sixes: int
    strike_rate: float
    average: float
    effective_sr: Optional[float] = None
    effective_avg: Optional[float] = None
    vulnerability: Optional[float] = None

class BaseStatsProcessor:
    """Base class for processing cricket statistics"""
    
    @staticmethod
    @lru_cache(maxsize=1024)
    def calculate_stats(runs: int, balls: int, dismissals: int) -> Tuple[float, float]:
        """Calculate strike rate and average with caching"""
        strike_rate = (runs / balls * 100) if balls > 0 else 0.0
        average = (runs / dismissals) if dismissals > 0 else float('inf')
        return strike_rate, average
    
    @staticmethod
    def extract_basic_stats(data: pd.DataFrame) -> BattingStats:
        """Extract basic batting statistics from a DataFrame"""
        runs = data['batruns'].sum()
        balls = data['ballfaced'].sum()
        dismissals = data['out'].sum()
        dot_balls = len(data[(data['score'] == 0) & (data['batruns'] == 0)])
        singles = len(data[data['batruns'] == 1])
        fours = len(data[data['batruns'] == 4])
        sixes = len(data[data['batruns'] == 6])
        
        strike_rate, average = BaseStatsProcessor.calculate_stats(runs, balls, dismissals)
        
        return BattingStats(
            runs=int(runs),
            balls=int(balls),
            dismissals=int(dismissals),
            dot_balls=int(dot_balls),
            singles=int(singles),
            fours=int(fours),
            sixes=int(sixes),
            strike_rate=strike_rate,
            average=average
        )
    
    @staticmethod
    def process_group(group_data: pd.DataFrame, min_balls: int = 5) -> Optional[BattingStats]:
        """Process a group of data and return stats if it meets minimum criteria"""
        if len(group_data) < min_balls:
            return None
            
        return BaseStatsProcessor.extract_basic_stats(group_data)
        
    @staticmethod
    def calculate_vulnerability(stats: BattingStats, phase: Optional[int] = None) -> float:
        """Calculate vulnerability score with phase-specific adjustments"""
        # SR component
        if stats.balls >= 10 and stats.strike_rate > 0:
            sr_component = 100 / stats.strike_rate
        else:
            sr_component = 2.0  # Penalty for low sample/SR
            
        # Average component    
        if stats.average > 0 and stats.average != float('inf'):
            avg_component = 100 / stats.average
        else:
            avg_component = 0.5 * np.exp(-stats.balls/30)
            
        # Phase-specific weights
        sr_weight = 0.5
        avg_weight = 0.5
        if phase == 1:  # Powerplay
            sr_weight = 0.5
            avg_weight = 0.5
        elif phase == 3:  # Death
            sr_weight = 0.5
            avg_weight = 0.5
            
        # Calculate final score
        confidence = min(1.0, stats.balls / 30)
        raw_score = (sr_weight * sr_component + avg_weight * avg_component)
        vulnerability = raw_score * confidence * 20
        
        return min(100, vulnerability)