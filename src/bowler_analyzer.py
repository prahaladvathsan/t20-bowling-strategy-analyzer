import pandas as pd
import numpy as np
import json
from pathlib import Path
from data_processor import DataProcessor

class BowlerAnalyzer:
    """Analyzes bowler effectiveness with different lines and lengths"""
    
    def __init__(self, data=None):
        """Initialize with ball-by-ball dataset or load from saved profiles"""
        if data is not None:
            self.data = data
            self.bowler_profiles = self._create_bowler_profiles()
        else:
            # Load from saved profiles
            self.bowler_profiles = self._load_profiles()
            self.data = None  # No need to keep data in memory if loading from file
    
    def _load_profiles(self):
        """Load bowler profiles from saved JSON file"""
        try:
            db_path = Path(__file__).parent.parent / "db"
            with open(db_path / "bowler_profiles.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise ValueError("No saved profiles found. Please run the backend processor first.")

    def _create_bowler_profiles(self):
        """Create profiles for bowlers based on their performance"""
        profiles = {}
        
        for bowler in self.data['bowl'].unique():
            bowler_data = self.data[self.data['bowl'] == bowler]
            
            # Get bowler type
            bowl_kind = bowler_data['bowl_kind'].iloc[0] if 'bowl_kind' in bowler_data.columns else 'Unknown'
            bowl_style = bowler_data['bowl_style'].iloc[0] if 'bowl_style' in bowler_data.columns else 'Unknown'
            
            # Calculate overall stats
            total_runs = bowler_data['score'].sum()
            total_balls = len(bowler_data)
            wickets = bowler_data['out'].sum()
            
            # Use utility functions for calculations
            economy = DataProcessor.calculate_economy(total_runs, total_balls)
            average = DataProcessor.calculate_average(total_runs, wickets)
            bowling_strike_rate = DataProcessor.calculate_bowling_strike_rate(total_balls, wickets)
            
            # Analysis by line and length
            line_length_stats = {}
            
            if 'line' in bowler_data.columns and 'length' in bowler_data.columns:
                for (line, length), ll_data in bowler_data.groupby(['line', 'length']):
                    if pd.isna(line) or pd.isna(length):
                        continue
                        
                    runs = ll_data['score'].sum()
                    balls = ll_data['score'].count()
                    wickets = ll_data['out'].sum()
                    
                    if balls >= 1:
                        line_display = DataProcessor.LINE_DISPLAY.get(int(line), 'Unknown')
                        length_display = DataProcessor.LENGTH_DISPLAY.get(int(length), 'Unknown')
                        
                        line_length_stats[(line_display, length_display)] = {
                            'runs': int(runs),
                            'balls': int(balls),
                            'wickets': int(wickets),
                            'economy': DataProcessor.calculate_economy(runs, balls),
                            'average': DataProcessor.calculate_average(runs, wickets),
                            'bowling_strike_rate': (balls/wickets) if wickets > 0 else float('inf')
                        }
            
            profiles[bowler] = {
                'bowl_kind': bowl_kind,
                'bowl_style': bowl_style,
                'total_runs': total_runs,
                'total_balls': total_balls,
                'wickets': wickets,
                'economy': economy,
                'average': average,
                'bowling_strike_rate': bowling_strike_rate,
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
        for combo_key, stats in profile['by_line_length'].items():
            # Skip combinations with too few deliveries
            if stats['balls'] < 5:
                continue
                
            # Parse the display names
            combo_str = combo_key.strip("()")
            line_display, length_display = [s.strip().strip("'") for s in combo_str.split(',')]
                
            # Calculate effectiveness score (lower is better)
            if stats['wickets'] > 0:
                effectiveness = (0.7 * stats['economy']) - (0.3 * (120 / stats['bowling_strike_rate']))
            else:
                effectiveness = stats['economy']
            
            recommendations.append({
                'line': line_display,
                'length': length_display,
                'effectiveness': effectiveness,
                'economy': stats['economy'],
                'strike_rate': stats['bowling_strike_rate'] if stats['wickets'] > 0 else float('inf'),
                'sample_size': stats['balls']
            })
        
        # Sort by effectiveness (lower is better)
        recommendations.sort(key=lambda x: x['effectiveness'])
        
        return recommendations[:3]  # Return top 3 recommendations