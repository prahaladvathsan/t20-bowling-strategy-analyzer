import pandas as pd
import numpy as np
from src.data_processor import DataProcessor

class BowlerAnalyzer:
    """Analyzes bowler effectiveness with different lines and lengths"""
    
    def __init__(self, data):
        """Initialize with ball-by-ball dataset"""
        self.data = data
        self.bowler_profiles = self._create_bowler_profiles()
        
    def _create_bowler_profiles(self):
        """Create profiles for bowlers based on their performance"""
        profiles = {}
        
        # Group by bowler
        bowlers = self.data['bowl'].unique()
        
        for bowler in bowlers:
            bowler_data = self.data[self.data['bowl'] == bowler]
            
            # Get bowler type
            try:
                bowl_kind = bowler_data['bowl_kind'].iloc[0]
                bowl_style = bowler_data['bowl_style'].iloc[0] if 'bowl_style' in bowler_data.columns else 'Unknown'
            except:
                bowl_kind = 'Unknown'
                bowl_style = 'Unknown'
            
            # Calculate overall stats
            total_runs = bowler_data['score'].sum()
            total_balls = len(bowler_data)
            wickets = bowler_data['out'].sum()
            
            # Economy and average
            economy = (total_runs / total_balls * 6) if total_balls > 0 else 0
            average = (total_runs / wickets) if wickets > 0 else float('inf')
            strike_rate = (total_balls / wickets) if wickets > 0 else float('inf')
            
            # Analysis by line and length
            line_length_stats = {}
            
            # Group by line and length
            if 'line' in bowler_data.columns and 'length' in bowler_data.columns:
                ll_stats = (bowler_data
                    .groupby(['line', 'length'], as_index=False)
                    .agg({
                        'score': 'sum',
                        'out': 'sum'
                    }).assign(balls=lambda x: 1)
                )
                
                for _, row in ll_stats.iterrows():
                    line = row['line']  # Already numeric 0-4
                    length = row['length']  # Already numeric 0-5
                    key = (line, length)
                    
                    runs = row['score']
                    balls = row['balls']
                    outs = row['out']
                    
                    eco = (runs / balls * 6) if balls > 0 else 0
                    avg = (runs / outs) if outs > 0 else float('inf')
                    sr = (balls / outs) if outs > 0 else float('inf')
                    
                    line_length_stats[key] = {
                        'runs': int(runs),
                        'balls': int(balls),
                        'wickets': int(outs),
                        'economy': eco,
                        'average': avg,
                        'strike_rate': sr
                    }
            
            # Store profile
            profiles[bowler] = {
                'bowl_kind': bowl_kind,
                'bowl_style': bowl_style,
                'total_runs': total_runs,
                'total_balls': total_balls,
                'wickets': wickets,
                'economy': economy,
                'average': average,
                'strike_rate': strike_rate,
                'by_line_length': line_length_stats
            }
        
        return profiles
    
    def get_bowler_profile(self, bowler):
        """Return profile for a specific bowler"""
        if bowler in self.bowler_profiles:
            return self.bowler_profiles[bowler]
        return None
    
    def get_optimal_line_length(self, bowler):
        """Get optimal line and length combinations for a bowler"""
        profile = self.get_bowler_profile(bowler)
        if not profile or not profile['by_line_length']:
            return []
        
        # Calculate effectiveness scores
        recommendations = []
        for (line, length), stats in profile['by_line_length'].items():
            # Skip combinations with too few deliveries
            if stats['balls'] < 5:
                continue
                
            # Calculate effectiveness score (lower is better)
            if stats['wickets'] > 0:
                effectiveness = (0.7 * stats['economy']) - (0.3 * (120 / stats['strike_rate']))
            else:
                effectiveness = stats['economy']
            
            # Use display mappings for line and length
            line_display = DataProcessor.LINE_DISPLAY.get(int(line), 'Unknown')
            length_display = DataProcessor.LENGTH_DISPLAY.get(int(length), 'Unknown')
            
            recommendations.append({
                'line': line_display,
                'length': length_display,
                'effectiveness': effectiveness,
                'economy': stats['economy'],
                'strike_rate': stats['strike_rate'] if stats['wickets'] > 0 else float('inf'),
                'sample_size': stats['balls']
            })
        
        # Sort by effectiveness (lower is better)
        recommendations.sort(key=lambda x: x['effectiveness'])
        
        return recommendations[:3]  # Return top 3 recommendations