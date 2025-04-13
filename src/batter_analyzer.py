import pandas as pd
import numpy as np
from src.data_processor import DataProcessor

class BatterVulnerabilityAnalyzer:
    """Analyzes batter's weaknesses against specific bowling types"""
    
    def __init__(self, data):
        """Initialize with ball-by-ball dataset"""
        self.data = data.copy()
        
        # Check for critical columns
        required_cols = ['bat', 'score', 'out']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Pre-calculate profiles
        self.batter_profiles = self._create_batter_profiles()
        
    def _create_batter_profiles(self):
        """Create profiles for batters based on their performance"""
        profiles = {}
        
        for batter, batter_data in self.data.groupby('bat'):
            try:
                # Basic stats
                total_runs = batter_data['score'].sum()
                total_balls = len(batter_data)
                dismissals = batter_data['out'].sum()
                
                # Get batting hand if available
                bat_hand = batter_data['bat_hand'].mode().iloc[0] if 'bat_hand' in batter_data.columns else "Unknown"
                
                # Use utility functions for calculations
                strike_rate = DataProcessor.calculate_strike_rate(total_runs, total_balls)
                average = DataProcessor.calculate_average(total_runs, dismissals)
                
                # Process phase data
                phase_stats = self._process_phase_data(batter_data)
                
                # Process bowling style data
                bowl_style_dict = self._process_bowling_style_data(batter_data)
                
                # Process line and length data
                line_length_stats = self._process_line_length_data(batter_data)
                
                # Store the complete profile
                profiles[batter] = {
                    'bat_hand': bat_hand,
                    'total_runs': int(total_runs),
                    'total_balls': int(total_balls),
                    'dismissals': int(dismissals),
                    'strike_rate': strike_rate,
                    'average': average,
                    'by_phase': phase_stats,
                    'vs_bowler_styles': bowl_style_dict,
                    'vs_line_length': line_length_stats
                }
                
            except Exception as e:
                print(f"Error processing batter {batter}: {str(e)}")
                continue
        
        return profiles

    def _process_phase_data(self, batter_data):
        """Process phase-specific data for a batter"""
        phase_stats = {}
        
        if 'phase' in batter_data.columns:
            for phase, phase_data in batter_data.groupby('phase'):
                if len(phase_data) > 0:
                    runs = phase_data['score'].sum()
                    balls = len(phase_data)
                    outs = phase_data['out'].sum()
                    
                    phase_stats[phase] = {
                        'runs': int(runs),
                        'balls': int(balls),
                        'dismissals': int(outs),
                        'strike_rate': DataProcessor.calculate_strike_rate(runs, balls),
                        'average': DataProcessor.calculate_average(runs, outs)
                    }
        
        return phase_stats if phase_stats else None
    
    def _process_bowling_style_data(self, batter_data):
        """Process bowling style matchup data for a batter"""
        bowl_style_dict = {}
        
        if 'bowl_style' in batter_data.columns:
            for style in batter_data['bowl_style'].unique():
                if pd.isna(style) or style == '-' or style == '':
                    continue
                
                style_data = batter_data[batter_data['bowl_style'] == style]
                runs = style_data['score'].sum()
                balls = len(style_data)
                outs = style_data['out'].sum()
                
                if balls >= 3:
                    bowl_style_dict[style] = {
                        'runs': int(runs),
                        'balls': int(balls),
                        'dismissals': int(outs),
                        'strike_rate': DataProcessor.calculate_strike_rate(runs, balls),
                        'average': DataProcessor.calculate_average(runs, outs)
                    }
        
        return bowl_style_dict if bowl_style_dict else None
    
    def _process_line_length_data(self, batter_data):
        """Process line and length data for a batter"""
        line_length_stats = {}
        
        if all(col in batter_data.columns for col in ['line', 'length']):
            for (line, length), ll_data in batter_data.groupby(['line', 'length']):
                if pd.isna(line) or pd.isna(length):
                    continue
                    
                runs = ll_data['score'].sum()
                balls = len(ll_data)
                outs = ll_data['out'].sum()
                
                if balls >= 3:
                    line_display = DataProcessor.LINE_DISPLAY.get(int(line), 'Unknown')
                    length_display = DataProcessor.LENGTH_DISPLAY.get(int(length), 'Unknown')
                    
                    line_length_stats[(line_display, length_display)] = {
                        'runs': int(runs),
                        'balls': int(balls),
                        'dismissals': int(outs),
                        'strike_rate': DataProcessor.calculate_strike_rate(runs, balls),
                        'average': DataProcessor.calculate_average(runs, outs)
                    }
        
        return line_length_stats if line_length_stats else None
    
    def analyze_batter(self, batter):
        """Return complete analysis for a specific batter"""
        return self.batter_profiles.get(batter)
    
    def analyze_batter_by_phase(self, batter, phase=None):
        """Return phase-specific analysis for a batter"""
        profile = self.batter_profiles.get(batter)
        if not profile:
            return None
            
        # Return all phases if no specific phase requested
        if phase is None:
            return profile.get('by_phase')
        
        # Return specific phase data
        if profile.get('by_phase') and phase in profile['by_phase']:
            return profile['by_phase'][phase]
        
        return None
    
    def analyze_batter_vs_bowl_style(self, batter, style=None):
        """Return analysis of batter against specific bowling style"""
        profile = self.batter_profiles.get(batter)
        if not profile:
            return None
            
        # Return all styles if no specific style requested
        if style is None:
            return profile.get('vs_bowler_styles')
        
        # Return specific style data
        if profile.get('vs_bowler_styles') and style in profile['vs_bowler_styles']:
            return profile['vs_bowler_styles'][style]
        
        return None
    
    def analyze_batter_phase_vs_style(self, batter, phase=None, style=None):
        """Return phase-specific analysis against specific bowling style"""
        profile = self.batter_profiles.get(batter)
        if not profile or not profile.get('phase_bowl_style'):
            return None
            
        # Handle different cases based on what was requested
        if phase is None and style is None:
            # Return all phase-style data
            return profile.get('phase_bowl_style')
        elif phase is not None and style is None:
            # Return all styles for specific phase
            return profile.get('phase_bowl_style', {}).get(phase)
        elif phase is None and style is not None:
            # Return all phases for specific style
            result = {}
            for p, styles in profile.get('phase_bowl_style', {}).items():
                if style in styles:
                    result[p] = styles[style]
            return result if result else None
        else:
            # Return specific phase and style
            return profile.get('phase_bowl_style', {}).get(phase, {}).get(style)
    
    def get_available_bowl_styles(self):
        """Return a list of all bowling styles in the data"""
        if 'bowl_style' not in self.data.columns:
            return []
            
        # Get unique styles directly from the dataset
        styles = self.data['bowl_style'].dropna().unique()
        valid_styles = [s for s in styles if s != '-' and s != '']
        return sorted(valid_styles)