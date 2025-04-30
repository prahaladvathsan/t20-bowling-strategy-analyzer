"""Effective metrics calculator for cricket analytics"""
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from functools import lru_cache

class EffectiveMetricsCalculator:
    """Calculates effective strike rate and average metrics"""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize with full dataset"""
        self.data = data
        self._batter_id_cache: Dict[str, Optional[str]] = {}
        
    @lru_cache(maxsize=128)
    def _get_batter_id(self, batter: str) -> Optional[str]:
        """Get batter ID with caching"""
        if batter not in self._batter_id_cache:
            batter_data = self.data[self.data['bat'] == batter]
            if 'p_bat' in batter_data.columns and len(batter_data) > 0:
                self._batter_id_cache[batter] = batter_data['p_bat'].iloc[0] if not batter_data['p_bat'].isnull().all() else None
            else:
                self._batter_id_cache[batter] = None
        return self._batter_id_cache[batter]
        
    def _get_innings_period(self, batter: str, team_data: pd.DataFrame) -> Tuple[Optional[int], Optional[int], bool]:
        """Get the start and end index of batter's innings period"""
        batter_id = self._get_batter_id(batter)
        
        # Create batter presence mask
        presence_mask = (team_data['bat'] == batter)
        if batter_id is not None:
            presence_mask |= (team_data['p_bat'] == batter_id)
            if 'p_bat_ns' in team_data.columns:
                presence_mask |= (team_data['p_bat_ns'] == batter_id)
                
        batter_presence = team_data[presence_mask]
        if len(batter_presence) == 0:
            return None, None, False
            
        first_ball_idx = batter_presence.index.min()
        
        # Find dismissal
        dismissed = False
        if batter_id is not None and 'p_out' in team_data.columns:
            dismissal = team_data[team_data['p_out'] == batter_id]
            if len(dismissal) > 0:
                last_ball_idx = dismissal.index.max()
                dismissed = True
            else:
                dismissal = team_data[(team_data['bat'] == batter) & (team_data['out'] == True)]
                if len(dismissal) > 0:
                    last_ball_idx = dismissal.index.max()
                    dismissed = True
                else:
                    last_ball_idx = batter_presence.index.max()
        else:
            dismissal = team_data[(team_data['bat'] == batter) & (team_data['out'] == True)]
            if len(dismissal) > 0:
                last_ball_idx = dismissal.index.max()
                dismissed = True
            else:
                last_ball_idx = batter_presence.index.max()
                
        return first_ball_idx, last_ball_idx, dismissed
        
    def calculate_effective_metrics(self, batter: str) -> Tuple[Optional[float], Optional[float]]:
        """Calculate effective strike rate and average for a batter"""
        if 'p_match' not in self.data.columns or 'inns' not in self.data.columns:
            return None, None
            
        batter_id = self._get_batter_id(batter)
        if batter_id is None and len(self.data[self.data['bat'] == batter]) == 0:
            return None, None
            
        effective_sr_list = []
        effective_avg_list = []
        
        # Get all innings for this batter
        batter_mask = (self.data['bat'] == batter)
        if batter_id is not None:
            batter_mask |= (self.data['p_bat'] == batter_id)
            
        for (match_id, innings_id), _ in self.data[batter_mask].groupby(['p_match', 'inns']):
            innings_data = self.data[
                (self.data['p_match'] == match_id) & 
                (self.data['inns'] == innings_id)
            ]
            
            # Sort innings data
            if 'ball_id' in innings_data.columns:
                innings_data = innings_data.sort_values('ball_id')
            elif 'ball' in innings_data.columns:
                innings_data = innings_data.sort_values('ball')
                
            if 'team_bat' not in innings_data.columns:
                continue
                
            # Get team data
            batter_team = innings_data[
                (innings_data['bat'] == batter) | 
                (innings_data['p_bat'] == batter_id if batter_id is not None else False)
            ]
            if len(batter_team) == 0:
                continue
                
            team = batter_team['team_bat'].iloc[0]
            team_data = innings_data[innings_data['team_bat'] == team]
            
            # Get innings period
            first_ball_idx, last_ball_idx, was_dismissed = self._get_innings_period(batter, team_data)
            if first_ball_idx is None:
                continue
                
            # Calculate metrics for this period
            team_subset = team_data.loc[first_ball_idx:last_ball_idx]
            batter_on_strike = team_subset[team_subset['bat'] == batter]
            
            batter_runs = batter_on_strike['score'].sum()
            batter_balls = len(batter_on_strike)
            team_runs = team_subset['score'].sum()
            team_balls = len(team_subset)
            team_dismissals = team_subset['out'].sum()
            
            # Calculate effective strike rate
            if team_balls > 0 and batter_balls > 0 and team_runs > 0:
                batter_sr = (batter_runs / batter_balls) * 100
                team_sr = (team_runs / team_balls) * 100
                if team_sr > 0:
                    effective_sr_list.append(batter_sr / team_sr)
            
            # Calculate effective average
            if was_dismissed and team_dismissals > 0:
                team_avg = team_runs / team_dismissals
                if team_avg > 0:
                    effective_avg_list.append(batter_runs / team_avg)
        
        effective_sr = np.mean(effective_sr_list) if effective_sr_list else None
        effective_avg = np.mean(effective_avg_list) if effective_avg_list else None
        
        return effective_sr, effective_avg