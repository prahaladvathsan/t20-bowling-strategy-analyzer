"""Analyzes batter's performance and vulnerabilities"""
import pandas as pd
import numpy as np
import json
import math
from pathlib import Path
from data_processor import DataProcessor
from typing import Dict, List, Optional, Tuple, Union, Any

class BatterAnalyzer:
    """Analyzes batter's weaknesses against specific bowling types using a generalized pipeline"""
    
    # Analysis types
    ANALYSIS_TYPES = {
        'overall': {'group_by': None, 'min_balls': 10},
        'phase': {'group_by': 'phase', 'min_balls': 10},
        'bowling_style': {'group_by': 'bowl_style', 'min_balls': 3},
        'line_length': {'group_by': ['line', 'length'], 'min_balls': 1},
        'phase_line_length': {'group_by': ['phase', 'line', 'length'], 'min_balls': 1},
        'style_line_length': {'group_by': ['bowl_style', 'line', 'length'], 'min_balls': 1}
    }

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

    def _create_batter_profiles(self):
        """Create profiles for batters based on their performance"""
        profiles = {}
        
        for batter, batter_data in self.data.groupby('bat'):
            try:
                print(f"Processing batter: {batter}")
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
                        if pd.isna(phase) or phase == '-' or phase == '':
                            continue
                        phase_line_length_stats[phase] = self._process_line_length_data(phase_data, True)

                # Process bowler-style-wise line and length data
                style_line_length_stats = {}
                if 'bowl_style' in batter_data.columns:
                    for style, style_data in batter_data.groupby('bowl_style'):
                        if pd.isna(style) or style == '-' or style == '':
                            continue
                        style_line_length_stats[style] = self._process_line_length_data(style_data, True)

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

    def get_analysis_config(self, analysis_type: str, filters: Optional[Dict] = None) -> Dict:
        """
        Get configuration for a specific type of analysis
        
        Parameters:
            analysis_type: Type of analysis to perform
            filters: Optional filters to apply to the analysis
            
        Returns:
            Dictionary with analysis configuration
        """
        if analysis_type not in self.ANALYSIS_TYPES:
            raise ValueError(f"Invalid analysis type: {analysis_type}")
            
        config = self.ANALYSIS_TYPES[analysis_type].copy()
        
        # Add any filters
        if filters:
            config['filters'] = filters
            
        return config

    def process_analysis(self, data: pd.DataFrame, config: Dict) -> Dict:
        """
        Process data according to analysis configuration
        
        Parameters:
            data: DataFrame containing the data to analyze
            config: Analysis configuration dictionary
            
        Returns:
            Dictionary containing analysis results
        """
        # Apply filters if any
        if 'filters' in config:
            for col, value in config['filters'].items():
                data = data[data[col] == value]

        # For overall analysis
        if not config['group_by']:
            return self._process_grouped_data(data, None, config['min_balls'])

        # Handle multi-level grouping
        if isinstance(config['group_by'], list):
            # Create hierarchical groups
            groups = data.groupby(config['group_by'])
            result = {}
            
            for group_key, group_data in groups:
                # Create nested dictionary structure
                current_level = result
                for i, key in enumerate(group_key[:-1]):
                    if key not in current_level:
                        current_level[key] = {}
                    current_level = current_level[key]
                
                # Process the leaf level
                current_level[group_key[-1]] = self._process_grouped_data(
                    group_data, 
                    config['group_by'], 
                    config['min_balls']
                )
            
            return result
        
        # Single level grouping
        return self._process_grouped_data(data, config['group_by'], config['min_balls'])

    def analyze_batter_stats(self, 
                           batter: str, 
                           analysis_type: str, 
                           filters: Optional[Dict] = None, 
                           include_vulnerability: bool = True) -> Dict:
        """
        Unified analysis function for all batter statistics
        
        Parameters:
            batter: Name of the batter to analyze
            analysis_type: Type of analysis to perform
            filters: Optional filters to apply to the analysis
            include_vulnerability: Whether to include vulnerability metrics
            
        Returns:
            Dictionary containing analysis results
        """
        # Get batter profile or data
        if self.data is not None:
            batter_data = self.data[self.data['bat'] == batter]
        else:
            profile = self.batter_profiles.get(batter)
            if not profile:
                return None
            # Get the relevant section from saved profiles
            section_data = None
            if analysis_type == 'overall':
                section_data = profile
            elif analysis_type == 'phase':
                section_data = profile.get('by_phase', {})
            elif analysis_type == 'bowling_style':
                section_data = profile.get('vs_bowler_styles', {})
            elif analysis_type == 'line_length':
                section_data = profile.get('vs_line_length', {})
            elif analysis_type == 'phase_line_length':
                section_data = profile.get('phase_line_length', {})
            elif analysis_type == 'style_line_length':
                section_data = profile.get('style_line_length', {})
                
            # Apply filters if any
            if filters and section_data:
                filtered_data = {}
                if analysis_type == 'phase':
                    # For phase data, filter by phase number
                    phase = filters.get('phase')
                    if phase is not None and phase in section_data:
                        filtered_data[phase] = section_data[phase]
                    return filtered_data
                elif analysis_type == 'bowling_style':
                    # For bowling style data, filter by style
                    style = filters.get('bowl_style')
                    if style is not None and style in section_data:
                        filtered_data[style] = section_data[style]
                    return filtered_data
                elif analysis_type == 'phase_line_length':
                    # For phase-specific line/length data, filter by phase
                    phase = filters.get('phase')
                    if phase is not None and phase in section_data:
                        return section_data[phase]
                elif analysis_type == 'style_line_length':
                    # For style-specific line/length data, filter by bowling style
                    style = filters.get('bowl_style')
                    if style is not None and style in section_data:
                        return section_data[style]
                
                # If no filters matched or empty result, return the original section
                return section_data if not filtered_data else filtered_data
                
            return section_data
            
        # For live data analysis, use the pipeline
        config = self.get_analysis_config(analysis_type, filters)
        results = self.process_analysis(batter_data, config)
        
        return results

    def get_available_bowl_styles(self) -> List[str]:
        """Return list of available bowling styles"""
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

    def get_all_batters(self) -> List[str]:
        """Return list of all batters"""
        return list(self.batter_profiles.keys())

    def _process_grouped_data(self, data, group_by=None, min_balls=5):
        """Process statistics for a group of data"""
        result = {}
        
        # Handle case when the column(s) to group by don't exist
        if group_by and not all(col in data.columns for col in ([group_by] if isinstance(group_by, str) else group_by)):
            return result
        
        # Group the data
        groups = data.groupby(group_by) if group_by else [(None, data)]
        
        for group_key, group_data in groups:
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
            
            # Skip groups with insufficient data
            if balls < min_balls:
                continue
                
            # Calculate base stats
            strike_rate = DataProcessor.calculate_strike_rate(runs, balls)
            average = DataProcessor.calculate_average(runs, dismissals)
            
            # Calculate effective metrics
            effective_sr = None
            effective_avg = None
            
            if 'p_match' in group_data.columns and 'inns' in group_data.columns:
                team_stats = []
                for match_id, inns_data in group_data.groupby('p_match'):
                    for innings_id in inns_data['inns'].unique():
                        if 'team_bat' in inns_data.columns:
                            match_data = inns_data[inns_data['inns'] == innings_id]
                            if not match_data.empty:
                                team = match_data['team_bat'].iloc[0]
                                team_data = data[
                                    (data['p_match'] == match_id) & 
                                    (data['inns'] == innings_id) & 
                                    (data['team_bat'] == team)
                                ]
                                
                                # Get team data for period when batter was at crease
                                batter = match_data['bat'].iloc[0]
                                batter_id = match_data['p_bat'].iloc[0] if 'p_bat' in match_data.columns else None
                                
                                batter_presence = team_data[
                                    (team_data['bat'] == batter) | 
                                    (team_data['p_bat'] == batter_id if batter_id is not None else False) |
                                    (team_data['p_bat_ns'] == batter_id if 'p_bat_ns' in team_data.columns and batter_id is not None else False)
                                ]
                                
                                if len(batter_presence) > 0:
                                    first_ball_idx = batter_presence.index.min()
                                    
                                    # Find dismissal ball
                                    if batter_id is not None and 'p_out' in team_data.columns:
                                        dismissal_data = team_data[team_data['p_out'] == batter_id]
                                        if len(dismissal_data) > 0:
                                            last_ball_idx = dismissal_data.index.max()
                                        else:
                                            dismissal_by_name = team_data[(team_data['bat'] == batter) & (team_data['out'] == True)]
                                            if len(dismissal_by_name) > 0:
                                                last_ball_idx = dismissal_by_name.index.max()
                                            else:
                                                last_ball_idx = batter_presence.index.max()
                                    else:
                                        dismissal_by_name = team_data[(team_data['bat'] == batter) & (team_data['out'] == True)]
                                        if len(dismissal_by_name) > 0:
                                            last_ball_idx = dismissal_by_name.index.max()
                                        else:
                                            last_ball_idx = batter_presence.index.max()
                                    
                                    # Get team data for this period
                                    team_subset = team_data.loc[first_ball_idx:last_ball_idx]
                                    team_stats.append({
                                        'runs': team_subset['score'].sum(),
                                        'balls': len(team_subset),
                                        'dismissals': team_subset['out'].sum()
                                    })
                
                if team_stats:
                    team_runs = sum(s['runs'] for s in team_stats)
                    team_balls = sum(s['balls'] for s in team_stats)
                    team_dismissals = sum(s['dismissals'] for s in team_stats)
                    
                    if balls > 0 and team_balls > 0 and team_runs > 0:
                        batter_sr = (runs / balls) * 100
                        team_sr = (team_runs / team_balls) * 100
                        effective_sr = batter_sr / team_sr if team_sr > 0 else None
                        
                    if dismissals > 0 and team_dismissals > 0 and team_runs > 0:
                        batter_avg = runs / dismissals
                        team_avg = team_runs / team_dismissals
                        effective_avg = batter_avg / team_avg if team_avg > 0 else None
            
            # Create stats dictionary
            stats = {
                'runs': int(runs),
                'balls': int(balls),
                'dismissals': int(dismissals),
                'dot_balls': int(dot_balls),
                'strike_rate': strike_rate,
                'average': average
            }
            
            # Add effective metrics if available
            if effective_sr is not None:
                stats['effective_strike_rate'] = effective_sr
            if effective_avg is not None:
                stats['effective_average'] = effective_avg
            
            # Calculate and add vulnerability score
            vulnerability = self.calculate_vulnerability({
                'strike_rate': strike_rate,
                'average': average,
                'balls': balls,
                'dismissals': dismissals
            })
            stats['vulnerability'] = vulnerability
            
            # Add to result using appropriate key
            if isinstance(group_key, tuple) and 'line' in data.columns and 'length' in data.columns:
                # For line-length data, prepare display names
                if len(group_key) == 2:  # Basic line-length grouping
                    line_display = DataProcessor.LINE_DISPLAY.get(int(group_key[0]), 'Unknown')
                    length_display = DataProcessor.LENGTH_DISPLAY.get(int(group_key[1]), 'Unknown')
                    result[(line_display, length_display)] = stats
                else:  # Other multi-level groupings
                    result[group_key] = stats
            else:
                result[group_key] = stats
        
        return result

    def calculate_vulnerability(self, stats: Dict) -> float:
        """Calculate vulnerability score"""
        sr_component = 0
        avg_component = 0
        
        # Strike rate component (batting efficiency)
        balls_faced = stats.get('total_balls', stats.get('balls', 0))  # Try total_balls first, then balls, default to 0
        if balls_faced >= 10:  # Minimum sample size
            if stats['strike_rate'] > 0:
                sr_component = 100 / stats['strike_rate']
            else:
                sr_component = 2.0  # High penalty for zero strike rate
        
        # Average component (dismissal tendency)
        if stats['average'] > 0:
            avg_component = 100 / stats['average']
        else:
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
            return None, None
            
        # Get batter ID if available
        if 'p_bat' in self.data.columns:
            batter_data = self.data[self.data['bat'] == batter]
            batter_id = batter_data['p_bat'].iloc[0] if len(batter_data) > 0 and not batter_data['p_bat'].isnull().all() else None
        else:
            batter_id = None
            
        if batter_id is None and len(self.data[self.data['bat'] == batter]) == 0:
            return None, None
            
        effective_sr_list = []
        effective_avg_list = []
        
        for (match_id, innings_id), _ in (self.data[
            (self.data['bat'] == batter) | 
            (self.data['p_bat'] == batter_id if batter_id is not None else False)
        ].groupby(['p_match', 'inns'])):
            
            innings_data = self.data[(self.data['p_match'] == match_id) & 
                                    (self.data['inns'] == innings_id)]
            
            if 'ball_id' in innings_data.columns:
                innings_data = innings_data.sort_values('ball_id')
            elif 'ball' in innings_data.columns:
                innings_data = innings_data.sort_values('ball')
                
            if 'team_bat' in innings_data.columns:
                batter_team_data = innings_data[
                    (innings_data['bat'] == batter) | 
                    (innings_data['p_bat'] == batter_id if batter_id is not None else False)
                ]
                if len(batter_team_data) == 0:
                    continue
                    
                team = batter_team_data['team_bat'].iloc[0]
                team_data = innings_data[innings_data['team_bat'] == team]
                
                batter_presence = team_data[
                    (team_data['bat'] == batter) | 
                    (team_data['p_bat'] == batter_id if batter_id is not None else False) |
                    (team_data['p_bat_ns'] == batter_id if 'p_bat_ns' in team_data.columns and batter_id is not None else False)
                ]
                
                if len(batter_presence) == 0:
                    continue
                
                first_ball_idx = batter_presence.index.min()
                
                if batter_id is not None and 'p_out' in team_data.columns:
                    dismissal_data = team_data[team_data['p_out'] == batter_id]
                    if len(dismissal_data) > 0:
                        last_ball_idx = dismissal_data.index.max()
                        batter_dismissed = True
                    else:
                        dismissal_by_name = team_data[(team_data['bat'] == batter) & (team_data['out'] == True)]
                        if len(dismissal_by_name) > 0:
                            last_ball_idx = dismissal_by_name.index.max()
                            batter_dismissed = True
                        else:
                            last_ball_idx = batter_presence.index.max()
                            batter_dismissed = False
                else:
                    dismissal_by_name = team_data[(team_data['bat'] == batter) & (team_data['out'] == True)]
                    if len(dismissal_by_name) > 0:
                        last_ball_idx = dismissal_by_name.index.max()
                        batter_dismissed = True
                    else:
                        last_ball_idx = batter_presence.index.max()
                        batter_dismissed = False
                
                team_subset = team_data.loc[first_ball_idx:last_ball_idx]
                batter_on_strike = team_subset[team_subset['bat'] == batter]
                batter_runs = batter_on_strike['score'].sum()
                batter_balls = len(batter_on_strike)
                
                team_runs = team_subset['score'].sum()
                team_balls = len(team_subset)
                team_dismissals = team_subset['out'].sum()
                
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
        
        eff_sr = np.mean(effective_sr_list) if effective_sr_list else None
        eff_avg = np.mean(effective_avg_list) if effective_avg_list else None
        
        return eff_sr, eff_avg

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
            min_balls=5
        )