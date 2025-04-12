import pandas as pd
import numpy as np

class BatterVulnerabilityAnalyzer:
    """Analyzes batter's weaknesses against specific bowling types"""
    
    def __init__(self, data):
        """Initialize with ball-by-ball dataset"""
        self.data = data
        self.batter_profiles = self._create_batter_profiles()
        
    def _create_batter_profiles(self):
        """Create profiles for batters based on their performance"""
        profiles = {}
        
        # Group by batter
        batters = self.data['bat'].unique()
        
        for batter in batters:
            batter_data = self.data[self.data['bat'] == batter]
            
            # Get batter hand
            try:
                bat_hand = batter_data['bat_hand'].iloc[0]
            except:
                bat_hand = 'Unknown'
            
            # Calculate overall stats
            total_runs = batter_data['score'].sum()
            total_balls = len(batter_data)
            dismissals = batter_data['out'].sum()
            
            # Strike rate and average
            strike_rate = (total_runs / total_balls * 100) if total_balls > 0 else 0
            average = (total_runs / dismissals) if dismissals > 0 else float('inf')
            
            # Analysis by bowling type
            bowl_kinds = batter_data.groupby('bowl_kind').agg({
                'score': ['sum', 'count'],
                'out': 'sum'
            })
            
            bowl_kind_stats = {}
            for kind, stats in bowl_kinds.iterrows():
                runs = stats[('score', 'sum')]
                balls = stats[('score', 'count')]
                outs = stats[('out', 'sum')]
                
                sr = (runs / balls * 100) if balls > 0 else 0
                avg = (runs / outs) if outs > 0 else float('inf')
                
                bowl_kind_stats[kind] = {
                    'runs': runs,
                    'balls': balls,
                    'dismissals': outs,
                    'strike_rate': sr,
                    'average': avg
                }
            
            # Store profile
            profiles[batter] = {
                'bat_hand': bat_hand,
                'total_runs': total_runs,
                'total_balls': total_balls,
                'dismissals': dismissals,
                'strike_rate': strike_rate,
                'average': average,
                'vs_bowler_types': bowl_kind_stats
                # Add more analysis here as needed
            }
        
        return profiles
    
    def analyze_batter(self, batter):
        """Return analysis for a specific batter"""
        if batter in self.batter_profiles:
            return self.batter_profiles[batter]
        return None