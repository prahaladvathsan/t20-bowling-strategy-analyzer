"""Bowling statistics processing utilities"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from functools import lru_cache

@dataclass
class BowlingStats:
    """Container for bowling statistics"""
    runs: int
    balls: int
    wickets: int
    dot_balls: int
    singles: int
    fours: int
    sixes: int
    no_balls: int
    wides: int
    leg_byes: int
    byes: int
    economy: float
    average: float
    bowling_strike_rate: float

class BowlerProcessor:
    """Processes bowling statistics"""
    
    @staticmethod
    @lru_cache(maxsize=1024)
    def calculate_bowling_stats(runs: int, balls: int, wickets: int) -> Tuple[float, float, float]:
        """Calculate bowling stats with caching"""
        economy = (runs / balls * 6) if balls > 0 else 0.0
        average = (runs / wickets) if wickets > 0 else float('inf')
        strike_rate = (balls / wickets) if wickets > 0 else float('inf')
        return economy, average, strike_rate
    
    @staticmethod
    def extract_bowling_stats(data: pd.DataFrame) -> BowlingStats:
        """Extract bowling statistics from a DataFrame"""
        runs = data['bowlruns'].sum()
        balls = data['ballfaced'].sum()
        wickets = data['out'].sum()
        
        # Count different types of deliveries
        dot_balls = len(data[data['score'] == 0])
        singles = len(data[data['score'] == 1])
        fours = len(data[data['score'] == 4])
        sixes = len(data[data['score'] == 6])
        
        # Count extras and byes
        no_balls = len(data[data['noball'] >= 1]) if 'noball' in data.columns else 0
        wides = len(data[data['wide'] >= 1]) if 'wide' in data.columns else 0
        leg_byes = len(data[data['legbyes'] >= 1]) if 'legbyes' in data.columns else 0
        byes = len(data[data['byes'] >= 1]) if 'byes' in data.columns else 0
        
        # Calculate rates
        economy, average, strike_rate = BowlerProcessor.calculate_bowling_stats(runs, balls, wickets)
        
        return BowlingStats(
            runs=int(runs),
            balls=int(balls),
            wickets=int(wickets),
            dot_balls=int(dot_balls),
            singles=int(singles),
            fours=int(fours),
            sixes=int(sixes),
            no_balls=int(no_balls),
            wides=int(wides),
            leg_byes=int(leg_byes),
            byes=int(byes),
            economy=economy,
            average=average,
            bowling_strike_rate=strike_rate
        )
    
    @staticmethod
    def process_group(group_data: pd.DataFrame, min_balls: int = 5) -> Optional[BowlingStats]:
        """Process a group of data and return stats if it meets minimum criteria"""
        if len(group_data) < min_balls:
            return None
            
        return BowlerProcessor.extract_bowling_stats(group_data)