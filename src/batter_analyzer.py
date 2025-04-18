import pandas as pd
import numpy as np
import json
import math
from pathlib import Path
# In batter_analyzer.py
from data_processor import DataProcessor

class BatterVulnerabilityAnalyzer:
    """Analyzes batter's weaknesses against specific bowling types"""
    
    def __init__(self, data=None):
        """Initialize with ball-by-ball dataset or load from saved profiles"""
        if data is not None:
            self.data = data.copy()
            
            # Check for critical columns
            required_cols = ['bat', 'score', 'out']
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Pre-calculate profiles
            self.batter_profiles = self._create_batter_profiles()
        else:
            # Load from saved profiles
            self.batter_profiles = self._load_profiles()
            self.data = None  # No need to keep data in memory if loading from file
    
    def _load_profiles(self):
        """Load batter profiles from saved JSON file"""
        try:
            db_path = Path(__file__).parent.parent / "db"
            with open(db_path / "batter_profiles.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise ValueError("No saved profiles found. Please run the backend processor first.")

    def calculate_vulnerability(self, stats):
        """
        Calculate a nuanced vulnerability score based on multiple factors:
        - Strike rate component (efficiency)
        - Average component (dismissal tendency) 
        - Dot ball component (pressure building ability)
        - Weights adjusted based on game phase
        - Confidence factor based on sample size
        
        Returns a score from 0-100 where higher values indicate greater vulnerability
        """
        # Base components
        sr_component = 0
        avg_component = 0
        
        # Strike rate component (batting efficiency)
        balls_faced = stats['balls']
        if balls_faced >= 10:  # Minimum sample size
            if stats['strike_rate'] > 0:
                sr_component = 100 / stats['strike_rate']
            else:
                sr_component = 2.0  # High penalty for zero strike rate
        
        # Average component (dismissal tendency)
        if stats['average'] > 0:
            avg_component = 100 / stats['average']
        else:
            # If no dismissals, use a function of balls faced
            avg_component = 0.5 * math.exp(-balls_faced/30)
        
        # Default weights
        weights = {
            'sr': 0.5,
            'avg': 0.5
        }
        
        # Calculate phase-specific weight adjustments if phase info is available
        phase = stats.get('phase', None)
        if phase == 1:  # Powerplay
            weights['sr'] = 0.5
            weights['avg'] = 0.5
        elif phase == 3:  # Death overs
            weights['sr'] = 0.5
            weights['avg'] = 0.5
        
        # Confidence factor based on sample size
        confidence = min(1.0, balls_faced / 30)
        
        # Calculate weighted vulnerability
        raw_vulnerability = (
            weights['sr'] * sr_component + 
            weights['avg'] * avg_component
        )
        
        # Scale by confidence and normalize to 0-100 range
        vulnerability = raw_vulnerability * confidence * 20
        
        # Cap at 100
        return min(100, vulnerability)

    def _calculate_effective_metrics(self, batter):
        """
        Calculate effective strike rate and effective average for a batter.
        
        Effective strike rate: 
            100*(Runs scored by batter/balls faced by batter)/(runs scored by team/balls faced by team)
            
        Effective average: 
            (Runs scored by batter/dismissals of batter)/(runs scored by team/dismissals of team)
            
        Both metrics are calculated from the first ball the batter faces until dismissal in every innings.
        Uses the p_bat_ns column to track batter presence as non-striker.
        """
        if 'p_match' not in self.data.columns or 'inns' not in self.data.columns:
            # Can't calculate innings-based metrics without required identifiers
            return None, None
            
        # Get batter ID if available
        if 'p_bat' in self.data.columns:
            # First, get all rows where this player is batting (using name)
            batter_data = self.data[self.data['bat'] == batter]
            
            # Extract their player ID
            if len(batter_data) > 0 and not batter_data['p_bat'].isnull().all():
                batter_id = batter_data['p_bat'].iloc[0]
            else:
                batter_id = None
        else:
            batter_id = None
            
        # Exit if we couldn't find any data
        if batter_id is None and len(self.data[self.data['bat'] == batter]) == 0:
            return None, None
            
        effective_sr_list = []
        effective_avg_list = []
        
        # Process each match and innings where batter participated
        for (match_id, innings_id), _ in (self.data[
            (self.data['bat'] == batter) | 
            (self.data['p_bat'] == batter_id if batter_id is not None else False)
        ].groupby(['p_match', 'inns'])):
            
            # Get all innings data
            innings_data = self.data[(self.data['p_match'] == match_id) & 
                                    (self.data['inns'] == innings_id)]
            
            # Sort by ball_id if available, otherwise by ball
            if 'ball_id' in innings_data.columns:
                innings_data = innings_data.sort_values('ball_id')
            elif 'ball' in innings_data.columns:
                innings_data = innings_data.sort_values('ball')
                
            # Get team of the batter
            if 'team_bat' in innings_data.columns:
                # Find the batter's team
                batter_team_data = innings_data[
                    (innings_data['bat'] == batter) | 
                    (innings_data['p_bat'] == batter_id if batter_id is not None else False)
                ]
                if len(batter_team_data) == 0:
                    continue
                    
                team = batter_team_data['team_bat'].iloc[0]
                
                # Get team data for this innings
                team_data = innings_data[innings_data['team_bat'] == team]
                
                # Find when batter is at the crease (either as striker or non-striker)
                batter_presence = team_data[
                    (team_data['bat'] == batter) | 
                    (team_data['p_bat'] == batter_id if batter_id is not None else False) |
                    (team_data['p_bat_ns'] == batter_id if 'p_bat_ns' in team_data.columns and batter_id is not None else False)
                ]
                
                if len(batter_presence) == 0:
                    continue
                
                # Find first appearance and dismissal/last appearance
                first_ball_idx = batter_presence.index.min()
                
                # Check for dismissal
                if batter_id is not None and 'p_out' in team_data.columns:
                    dismissal_data = team_data[team_data['p_out'] == batter_id]
                    if len(dismissal_data) > 0:
                        last_ball_idx = dismissal_data.index.max()
                        batter_dismissed = True
                    else:
                        # If no dismissal by ID, check if batter was out when on strike
                        dismissal_by_name = team_data[(team_data['bat'] == batter) & (team_data['out'] == True)]
                        if len(dismissal_by_name) > 0:
                            last_ball_idx = dismissal_by_name.index.max()
                            batter_dismissed = True
                        else:
                            last_ball_idx = batter_presence.index.max()
                            batter_dismissed = False
                else:
                    # Check if batter was out when on strike (by name)
                    dismissal_by_name = team_data[(team_data['bat'] == batter) & (team_data['out'] == True)]
                    if len(dismissal_by_name) > 0:
                        last_ball_idx = dismissal_by_name.index.max()
                        batter_dismissed = True
                    else:
                        last_ball_idx = batter_presence.index.max()
                        batter_dismissed = False
                
                # Get team data while batter was at crease
                team_subset = team_data.loc[first_ball_idx:last_ball_idx]
                
                # Calculate batter stats (only when on strike)
                batter_on_strike = team_subset[team_subset['bat'] == batter]
                batter_runs = batter_on_strike['score'].sum()
                batter_balls = len(batter_on_strike)
                
                # Calculate team stats for the whole period batter was at crease
                team_runs = team_subset['score'].sum()
                team_balls = len(team_subset)
                team_dismissals = team_subset['out'].sum()
                
                # Calculate effective metrics
                if team_balls > 0 and batter_balls > 0:
                    batter_sr = (batter_runs / batter_balls) * 100
                    team_sr = (team_runs / team_balls) * 100
                    
                    if team_sr > 0:
                        effective_sr = batter_sr / team_sr
                        effective_sr_list.append(effective_sr)
                
                if batter_dismissed and team_dismissals > 0:
                    team_avg = team_runs / team_dismissals
                    
                    if team_avg > 0:
                        effective_avg = batter_runs / team_avg
                        effective_avg_list.append(effective_avg)
        
        # Calculate average of all innings
        eff_sr = np.mean(effective_sr_list) if effective_sr_list else None
        eff_avg = np.mean(effective_avg_list) if effective_avg_list else None
        
        return eff_sr, eff_avg

    def _create_batter_profiles(self):
        """Create profiles for batters based on their performance"""
        profiles = {}
        
        for batter, batter_data in self.data.groupby('bat'):
            try:
                # Basic stats
                total_runs = batter_data['score'].sum()
                total_balls = len(batter_data)
                dismissals = batter_data['out'].sum()
                
                # Count dot balls
                dot_balls = len(batter_data[batter_data['score'] == 0])
                
                # Get batting hand if available
                bat_hand = batter_data['bat_hand'].mode().iloc[0] if 'bat_hand' in batter_data.columns else "Unknown"
                
                # Get batter ID if available
                batter_id = batter_data['p_bat'].mode().iloc[0] if 'p_bat' in batter_data.columns else None
                
                # Use utility functions for calculations
                strike_rate = DataProcessor.calculate_strike_rate(total_runs, total_balls)
                average = DataProcessor.calculate_average(total_runs, dismissals)
                
                # Calculate effective metrics
                effective_sr, effective_avg = self._calculate_effective_metrics(batter)
                
                # Process phase data
                phase_stats = self._process_phase_data(batter_data)
                
                # Process bowling style data
                bowl_style_dict = self._process_bowling_style_data(batter_data)
                
                # Process line and length data
                line_length_stats = self._process_line_length_data(batter_data)

                # Process phase-wise line and length data
                phase_line_length_stats = {}
                if 'phase' in batter_data.columns:
                    for phase, phase_data in batter_data.groupby('phase'):
                        phase_line_length_stats[phase] = self._process_line_length_data(phase_data, is_phase_analysis=True)

                # Process bowler-style-wise line and length data
                style_line_length_stats = {}
                if 'bowl_style' in batter_data.columns:
                    for style, style_data in batter_data.groupby('bowl_style'):
                        if pd.isna(style) or style == '-' or style == '':
                            continue
                        style_line_length_stats[style] = self._process_line_length_data(style_data, is_phase_analysis=True)  # Using lower threshold like phase analysis

                # Calculate overall vulnerability
                overall_stats = {
                    'runs': total_runs,
                    'balls': total_balls,
                    'dismissals': dismissals,
                    'strike_rate': strike_rate,
                    'average': average,
                    'dot_balls': dot_balls
                }
                
                vulnerability_score = self.calculate_vulnerability(overall_stats)
                
                # Store the complete profile
                profiles[batter] = {
                    'bat_hand': bat_hand,
                    'batter_id': batter_id,
                    'total_runs': int(total_runs),
                    'total_balls': int(total_balls),
                    'dismissals': int(dismissals),
                    'dot_balls': int(dot_balls),
                    'strike_rate': strike_rate,
                    'average': average,
                    'effective_strike_rate': effective_sr,
                    'effective_average': effective_avg,
                    'vulnerability': vulnerability_score,
                    'by_phase': phase_stats,
                    'vs_bowler_styles': bowl_style_dict,
                    'vs_line_length': line_length_stats,
                    'phase_line_length': phase_line_length_stats,
                    'style_line_length': style_line_length_stats
                }
                
            except Exception as e:
                print(f"Error processing batter {batter}: {str(e)}")
                continue
        
        return profiles

    def _process_phase_data(self, batter_data):
        """Process phase-specific data for a batter"""
        return self._process_grouped_data(batter_data, 'phase', min_balls=10)
    
    def _process_bowling_style_data(self, batter_data):
        """Process bowling style matchup data for a batter"""
        return self._process_grouped_data(batter_data, 'bowl_style', min_balls=3)
    
    def _process_line_length_data(self, batter_data, is_phase_analysis=False):
        """Process line and length data for a batter"""
        return self._process_grouped_data(
            batter_data, 
            ['line', 'length'], 
            min_balls=5,
            is_specialized_analysis=is_phase_analysis
        )
    
    def _process_grouped_data(self, data, group_by, min_balls=5, is_specialized_analysis=False):
        """
        Generalized function to process data grouped by specified columns
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The data to process
        group_by : str or list
            Column(s) to group by
        min_balls : int
            Minimum number of balls needed for a valid group
        is_specialized_analysis : bool
            If True, uses a lower threshold for filtering results
        
        Returns:
        --------
        dict
            Dictionary with processed statistics for each group
        """
        result = {}
        
        # Handle case when the column(s) to group by don't exist
        if not all(col in data.columns for col in ([group_by] if isinstance(group_by, str) else group_by)):
            return result
        
        # Group the data
        for group_key, group_data in data.groupby(group_by):
            # Skip if group key is invalid
            if isinstance(group_key, tuple):
                if any(pd.isna(k) or k == '-' or k == '' for k in group_key):
                    continue
            elif pd.isna(group_key) or group_key == '-' or group_key == '':
                continue
                
            # Calculate basic statistics
            runs = group_data['score'].sum()
            balls = len(group_data)
            dismissals = group_data['out'].sum()
            dot_balls = len(group_data[group_data['score'] == 0])
            
            # Calculate team statistics for effective metrics
            if 'p_match' in group_data.columns and 'inns' in group_data.columns:
                team_runs = 0
                team_balls = 0
                team_dismissals = 0
                
                for (match_id, innings_id), match_inns_data in group_data.groupby(['p_match', 'inns']):
                    if 'team_bat' in match_inns_data.columns:
                        team = match_inns_data['team_bat'].iloc[0]
                        team_data = data[
                            (data['p_match'] == match_id) & 
                            (data['inns'] == innings_id) & 
                            (data['team_bat'] == team)
                        ]
                        team_runs += team_data['score'].sum()
                        team_balls += len(team_data)
                        team_dismissals += team_data['out'].sum()
                
                # Calculate effective metrics with additional zero checks
                if balls > 0 and team_balls > 0 and team_runs > 0:
                    batter_sr = (runs / balls) * 100
                    team_sr = (team_runs / team_balls) * 100
                    effective_sr = batter_sr / team_sr if team_sr > 0 else None
                else:
                    effective_sr = None
                    
                if dismissals > 0 and team_dismissals > 0 and team_runs > 0:
                    batter_avg = runs / dismissals
                    team_avg = team_runs / team_dismissals
                    effective_avg = batter_avg / team_avg if team_avg > 0 else None
                else:
                    effective_avg = None
            else:
                effective_sr = None
                effective_avg = None
            
            # Use adjusted threshold if is_specialized_analysis
            if balls >= (1 if is_specialized_analysis else min_balls):
                # Process group_key to create standard dictionary key
                if isinstance(group_key, tuple):
                    # For line-length data, prepare display names
                    if len(group_key) == 2 and 'line' in data.columns and 'length' in data.columns:
                        line_display = DataProcessor.LINE_DISPLAY.get(int(group_key[0]), 'Unknown')
                        length_display = DataProcessor.LENGTH_DISPLAY.get(int(group_key[1]), 'Unknown')
                        dict_key = (line_display, length_display)
                    else:
                        dict_key = group_key
                else:
                    dict_key = group_key
                
                # Create statistics
                stats = {
                    'runs': int(runs),
                    'balls': int(balls),
                    'dismissals': int(dismissals),
                    'dot_balls': int(dot_balls),
                    'strike_rate': DataProcessor.calculate_strike_rate(runs, balls),
                    'average': DataProcessor.calculate_average(runs, dismissals)
                }
                
                # Add effective metrics if available
                if effective_sr is not None:
                    stats['effective_strike_rate'] = effective_sr
                if effective_avg is not None:
                    stats['effective_average'] = effective_avg
                
                # Add phase info if this is phase analysis
                if isinstance(group_by, str) and group_by == 'phase':
                    stats['phase'] = group_key
                
                # Calculate vulnerability
                stats['vulnerability'] = self.calculate_vulnerability(stats)
                
                # Add to result
                result[dict_key] = stats
        
        return result
    
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
    
    def analyze_style_line_length(self, batter, style=None):
        """Return style-specific line/length analysis for a batter"""
        profile = self.batter_profiles.get(batter)
        if not profile:
            return None
            
        # Return all styles if no specific style requested
        if style is None:
            return profile.get('style_line_length')
        
        # Return specific style data
        if profile.get('style_line_length') and style in profile['style_line_length']:
            return profile['style_line_length'][style]
        
        return None

    def get_available_bowl_styles(self):
        """Return a list of all bowling styles in the data"""
        # If we have live data, use that
        if self.data is not None and 'bowl_style' in self.data.columns:
            styles = self.data['bowl_style'].dropna().unique()
            valid_styles = [s for s in styles if s != '-' and s != '']
            return sorted(valid_styles)
            
        # Otherwise, extract styles from saved profiles
        all_styles = set()
        for profile in self.batter_profiles.values():
            if profile.get('vs_bowler_styles'):
                all_styles.update(profile['vs_bowler_styles'].keys())
        
        return sorted([s for s in all_styles if s != '-' and s != ''])
    
    def get_vulnerability_ranking(self, batter, category=None):
        """
        Return vulnerability ranking for a batter by different categories
        
        Parameters:
        - batter: Name of the batter to analyze
        - category: Optional category to rank (e.g., 'vs_bowler_styles', 'vs_line_length', 'by_phase')
                   If None, returns all categories
        
        Returns:
        - Dictionary with categories ranked by vulnerability
        """
        profile = self.batter_profiles.get(batter)
        if not profile:
            return None
            
        results = {}
        
        # Analyze by bowling styles
        if (category is None or category == 'vs_bowler_styles') and profile.get('vs_bowler_styles'):
            styles_ranked = sorted(
                [(style, data['vulnerability']) for style, data in profile['vs_bowler_styles'].items()],
                key=lambda x: x[1], 
                reverse=True
            )
            results['vs_bowler_styles'] = styles_ranked
            
        # Analyze by line/length
        if (category is None or category == 'vs_line_length') and profile.get('vs_line_length'):
            ll_ranked = sorted(
                [(f"{ll[0]}-{ll[1]}", data['vulnerability']) for ll, data in profile['vs_line_length'].items()],
                key=lambda x: x[1], 
                reverse=True
            )
            results['vs_line_length'] = ll_ranked
            
        # Analyze by game phase
        if (category is None or category == 'by_phase') and profile.get('by_phase'):
            phase_ranked = sorted(
                [(phase, data['vulnerability']) for phase, data in profile['by_phase'].items()],
                key=lambda x: x[1], 
                reverse=True
            )
            results['by_phase'] = phase_ranked
            
        return results if results else None

    def get_all_batters(self):
        """Return a list of all batters in the profiles"""
        return list(self.batter_profiles.keys())

    def get_data_columns(self):
        """Return a list of columns in the data"""
        if self.data is not None:
            return list(self.data.keys())
        return []  # Return empty list if no data is loaded

    def analyze_bowling_styles(self, batter):
        """Analyze batter's performance against different bowling styles"""
        bowl_style_data = self.analyze_batter_vs_bowl_style(batter)
        if not bowl_style_data:
            return None
            
        style_analysis = []
        for style, stats in bowl_style_data.items():
            style_analysis.append({
                'style': style,
                'strike_rate': stats['strike_rate'],
                'average': stats['average'],
                'balls': stats['balls'],
                'dismissals': stats['dismissals'],
                'vulnerability': stats['vulnerability']
            })
        
        return style_analysis