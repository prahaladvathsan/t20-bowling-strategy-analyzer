import pandas as pd
import numpy as np
from src.data_processor import DataProcessor

class BatterVulnerabilityAnalyzer:
    """Analyzes batter's weaknesses against specific bowling types"""
    
    def __init__(self, data):
        """Initialize with ball-by-ball dataset"""
        # Make a copy of the data to avoid modifying the original
        self.data = data.copy()
        
        # Import display mappings
        self.line_display = DataProcessor.LINE_DISPLAY
        self.length_display = DataProcessor.LENGTH_DISPLAY
        
        # Print columns in dataset for debugging
        print(f"Columns in dataset: {list(self.data.columns)}")
        
        # Check for critical columns
        required_cols = ['bat', 'score', 'out']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Ensure we have bowling style information
        if 'bowl_style' not in self.data.columns:
            print("WARNING: 'bowl_style' column not found in dataset. This will limit analysis capabilities.")
            
        # Pre-calculate profiles
        self.batter_profiles = self._create_batter_profiles()
        
    def _create_batter_profiles(self):
        """Create profiles for batters based on their performance"""
        profiles = {}
        
        # Process each batter
        for batter, batter_data in self.data.groupby('bat'):
            try:
                print(f"Processing batter: {batter}")
                
                # Basic stats
                total_runs = batter_data['score'].sum()
                total_balls = len(batter_data)
                dismissals = batter_data['out'].sum()
                
                # Get batting hand if available
                bat_hand = "Unknown"
                if 'bat_hand' in batter_data.columns and not batter_data['bat_hand'].isna().all():
                    bat_hand = batter_data['bat_hand'].mode().iloc[0]
                
                # Calculate strike rate and average
                strike_rate = (total_runs / total_balls * 100) if total_balls > 0 else 0
                average = (total_runs / dismissals) if dismissals > 0 else float('inf')
                
                # Initialize storage dictionaries
                phase_stats = {}
                bowl_kind_dict = {}
                bowl_style_dict = {}
                line_length_stats = {}
                phase_bowl_style_stats = {}
                
                # Process phase data if available
                if 'phase' in batter_data.columns:
                    for phase, phase_data in batter_data.groupby('phase'):
                        if len(phase_data) > 0:
                            runs = phase_data['score'].sum()
                            balls = len(phase_data)
                            outs = phase_data['out'].sum()
                            sr = (runs / balls * 100) if balls > 0 else 0
                            avg = (runs / outs) if outs > 0 else float('inf')
                            
                            phase_stats[phase] = {
                                'runs': int(runs),
                                'balls': int(balls),
                                'dismissals': int(outs),
                                'strike_rate': sr,
                                'average': avg
                            }
                
                # Process bowling kind data if available
                if 'bowl_kind' in batter_data.columns:
                    for bowl_kind, kind_data in batter_data.groupby('bowl_kind'):
                        if pd.isna(bowl_kind) or bowl_kind == '':
                            continue
                            
                        runs = kind_data['score'].sum()
                        balls = len(kind_data)
                        outs = kind_data['out'].sum()
                        sr = (runs / balls * 100) if balls > 0 else 0
                        avg = (runs / outs) if outs > 0 else float('inf')
                        
                        bowl_kind_dict[bowl_kind] = {
                            'runs': int(runs),
                            'balls': int(balls),
                            'dismissals': int(outs),
                            'strike_rate': sr,
                            'average': avg
                        }
                
                # Process bowling style data manually to ensure it works
                if 'bowl_style' in batter_data.columns:
                    print(f"Processing bowling styles for {batter}")
                    unique_styles = batter_data['bowl_style'].unique()
                    print(f"  Unique styles: {unique_styles}")
                    
                    for style in unique_styles:
                        # Skip empty or invalid styles
                        if pd.isna(style) or style == '-' or style == '':
                            continue
                        
                        # Get data for this style
                        style_data = batter_data[batter_data['bowl_style'] == style]
                        
                        runs = style_data['score'].sum()
                        balls = len(style_data)
                        outs = style_data['out'].sum()
                        
                        # Only include if we have reasonable sample size
                        if balls >= 3:
                            sr = (runs / balls * 100) if balls > 0 else 0
                            avg = (runs / outs) if outs > 0 else float('inf')
                            
                            print(f"  Adding style {style}: {balls} balls, {runs} runs, SR {sr:.2f}")
                            
                            bowl_style_dict[style] = {
                                'runs': int(runs),
                                'balls': int(balls),
                                'dismissals': int(outs),
                                'strike_rate': sr,
                                'average': avg
                            }
                
                    print(f"  Total styles added: {len(bowl_style_dict)}")
                
                # Process line and length data if available
                if all(col in batter_data.columns for col in ['line', 'length']):
                    for (line, length), ll_data in batter_data.groupby(['line', 'length']):
                        if pd.isna(line) or pd.isna(length):
                            continue
                            
                        runs = ll_data['score'].sum()
                        balls = len(ll_data)
                        outs = ll_data['out'].sum()
                        
                        if balls >= 3:
                            sr = (runs / balls * 100) if balls > 0 else 0
                            avg = (runs / outs) if outs > 0 else float('inf')
                            
                            # Convert line and length to display values
                            line_display = self.line_display.get(int(line), 'Unknown')
                            length_display = self.length_display.get(int(length), 'Unknown')
                            
                            line_length_stats[(line_display, length_display)] = {
                                'runs': int(runs),
                                'balls': int(balls),
                                'dismissals': int(outs),
                                'strike_rate': sr,
                                'average': avg
                            }
                
                # Process phase-specific bowling style data if both columns exist
                if all(col in batter_data.columns for col in ['phase', 'bowl_style']):
                    for phase, phase_data in batter_data.groupby('phase'):
                        if len(phase_data) < 5:
                            continue
                            
                        phase_bowl_style_stats[phase] = {}
                        
                        for style in phase_data['bowl_style'].unique():
                            if pd.isna(style) or style == '-' or style == '':
                                continue
                                
                            style_phase_data = phase_data[phase_data['bowl_style'] == style]
                            
                            runs = style_phase_data['score'].sum()
                            balls = len(style_phase_data)
                            outs = style_phase_data['out'].sum()
                            
                            if balls >= 3:
                                sr = (runs / balls * 100) if balls > 0 else 0
                                avg = (runs / outs) if outs > 0 else float('inf')
                                
                                phase_bowl_style_stats[phase][style] = {
                                    'runs': int(runs),
                                    'balls': int(balls),
                                    'dismissals': int(outs),
                                    'strike_rate': sr,
                                    'average': avg
                                }
                
                # Store the complete profile
                profiles[batter] = {
                    'bat_hand': bat_hand,
                    'total_runs': int(total_runs),
                    'total_balls': int(total_balls),
                    'dismissals': int(dismissals),
                    'strike_rate': strike_rate,
                    'average': average,
                    'by_phase': phase_stats if phase_stats else None,
                    'vs_bowler_types': bowl_kind_dict if bowl_kind_dict else None,
                    'vs_bowler_styles': bowl_style_dict if bowl_style_dict else None,
                    'vs_line_length': line_length_stats if line_length_stats else None,
                    'phase_bowl_style': phase_bowl_style_stats if phase_bowl_style_stats else None
                }
                
                # Debug output
                # if 'vs_bowler_styles' in profiles[batter]:
                #    print(f"Final vs_bowler_styles for {batter} has {len(profiles[batter]['vs_bowler_styles'] or {})} entries")
                
            except Exception as e:
                print(f"Error processing batter {batter}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        return profiles
    
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