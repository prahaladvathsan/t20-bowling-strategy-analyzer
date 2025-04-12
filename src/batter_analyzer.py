import pandas as pd
import numpy as np

class BatterVulnerabilityAnalyzer:
    """Analyzes batter's weaknesses against specific bowling types"""
    
    def __init__(self, data):
        """Initialize with ball-by-ball dataset"""
        # Only keep necessary columns
        required_cols = ['bat', 'bat_hand', 'score', 'out', 'bowl_kind']
        self.data = data[required_cols].copy()
        
        # Pre-calculate profiles for better performance
        self.batter_profiles = self._create_batter_profiles()
        
    def _create_batter_profiles(self):
        """Create profiles for batters based on their performance"""
        profiles = {}
        
        # First group by batter to process each one
        for batter, batter_data in self.data.groupby('bat'):
            try:
                # Basic stats calculation
                total_runs = batter_data['score'].sum()
                total_balls = len(batter_data)
                dismissals = batter_data['out'].sum()
                
                # Get batter hand (take most common value)
                bat_hand = batter_data['bat_hand'].mode().iloc[0] if not batter_data['bat_hand'].isna().all() else 'Unknown'
                
                # Calculate strike rate and average
                strike_rate = (total_runs / total_balls * 100) if total_balls > 0 else 0
                average = (total_runs / dismissals) if dismissals > 0 else float('inf')
                
                # Analysis by bowling type (using more efficient aggregation)
                bowl_kind_stats = (
                    batter_data.groupby('bowl_kind')
                    .agg({
                        'score': ['sum', 'size'],
                        'out': 'sum'
                    })
                    .reset_index()
                )
                
                bowl_kind_dict = {}
                for _, row in bowl_kind_stats.iterrows():
                    bowl_type = row['bowl_kind']
                    runs = row[('score', 'sum')]
                    balls = row[('score', 'size')]
                    outs = row[('out', 'sum')]
                    
                    sr = (runs / balls * 100) if balls > 0 else 0
                    avg = (runs / outs) if outs > 0 else float('inf')
                    
                    bowl_kind_dict[bowl_type] = {
                        'runs': int(runs),  # Convert to int to save memory
                        'balls': int(balls),
                        'dismissals': int(outs),
                        'strike_rate': sr,
                        'average': avg
                    }
                
                # Store profile with minimal data
                profiles[batter] = {
                    'bat_hand': bat_hand,
                    'total_runs': int(total_runs),
                    'total_balls': int(total_balls),
                    'dismissals': int(dismissals),
                    'strike_rate': strike_rate,
                    'average': average,
                    'vs_bowler_types': bowl_kind_dict
                }
                
            except Exception as e:
                print(f"Error processing batter {batter}: {str(e)}")
                continue
        
        return profiles
    
    def analyze_batter(self, batter):
        """Return analysis for a specific batter"""
        return self.batter_profiles.get(batter)