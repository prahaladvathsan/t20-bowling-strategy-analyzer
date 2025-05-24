"""Analyzes batter's performance and vulnerabilities"""
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
import json
from pathlib import Path
from functools import lru_cache
from stats_processor import BaseStatsProcessor, BattingStats
from effective_metrics import EffectiveMetricsCalculator
from data_processor import DataFrameProcessor, GroupConfig

class BatterAnalyzer:
    """Analyzes batter's weaknesses against specific bowling types using a generalized pipeline"""
    
    # Analysis configurations
    ANALYSIS_CONFIGS = {
        'overall': GroupConfig(group_by=None, min_balls=10),
        'phase': GroupConfig(group_by='phase', min_balls=10),
        'bowling_style': GroupConfig(group_by='bowl_style', min_balls=3),
        'batting_hand': GroupConfig(group_by='bat_hand', min_balls=5),
        'bat_pos': GroupConfig(group_by='bat_pos', min_balls=5),
        'line_length': GroupConfig(group_by=['line', 'length'], min_balls=1),
        'phase_line_length': GroupConfig(group_by=['phase', 'line', 'length'], min_balls=1),
        'style_line_length': GroupConfig(group_by=['bowl_style', 'line', 'length'], min_balls=1)
    }

    def __init__(self, data: Optional[pd.DataFrame] = None):
        """Initialize with ball-by-ball dataset or load from saved profiles"""
        if data is not None:
            self.data = data.copy()
            
            # Validate critical columns
            missing_cols = DataFrameProcessor.validate_columns(self.data, ['bat', 'score', 'out'])
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Initialize processors
            self.stats_processor = BaseStatsProcessor()
            self.metrics_calculator = EffectiveMetricsCalculator(self.data)
            
            # Pre-calculate profiles
            self.batter_profiles = self._create_batter_profiles()
        else:
            # Load from saved profiles
            self.batter_profiles = self._load_profiles()
            self.data = None

    def _load_profiles(self) -> Dict:
        """Load batter profiles from saved JSON file"""
        try:
            db_path = Path(__file__).parent.parent / "db"
            with open(db_path / "batter_profiles.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise ValueError("No saved profiles found. Please run the backend processor first.")

    @lru_cache(maxsize=128)
    def get_batter_name(self, batter_id: str) -> Optional[str]:
        """Get batter name with caching"""
        if self.data is None:
            return None
            
        batter_data = self.data[self.data['p_bat'] == batter_id]
        if 'bat' in batter_data.columns and len(batter_data) > 0:
            return batter_data['bat'].iloc[0] if not batter_data['bat'].isnull().all() else None
        return None

    def _create_batter_profiles(self) -> Dict:
        """Create profiles for batters based on their performance"""
        profiles = {}
        
        for batter_id, batter_data in self.data.groupby('p_bat'):
            try:
                # Filter out rows with unknown values
                batter_data = batter_data[
                    (batter_data['bat_hand'] != 'unknown') &
                    (batter_data['bowl_style'] != 'unknown') &
                    (batter_data['line'] != 'unknown') &
                    (batter_data['length'] != 'unknown')
                ]
                
                if len(batter_data) == 0:
                    continue
                
                # Get basic stats
                stats = self.stats_processor.extract_basic_stats(batter_data)
                
                # Get batting hand and name
                bat_hand = batter_data['bat_hand'].mode().iloc[0]
                batter = self.get_batter_name(batter_id)
                
                # Calculate effective metrics
                effective_sr, effective_avg = self.metrics_calculator.calculate_effective_metrics(batter)
                
                # Process different analysis types
                phase_stats = self._process_analysis(batter_data, 'phase')
                bowl_style_stats = self._process_analysis(batter_data, 'bowling_style')
                batting_hand_stats = self._process_analysis(batter_data, 'batting_hand')
                bat_pos_stats = self._process_analysis(batter_data, 'bat_pos')
                line_length_stats = self._process_analysis(batter_data, 'line_length')
                phase_ll_stats = self._process_analysis(batter_data, 'phase_line_length')
                style_ll_stats = self._process_analysis(batter_data, 'style_line_length')
                
                # Calculate vulnerability
                vulnerability = self.stats_processor.calculate_vulnerability(stats)
                
                # Create profile
                profiles[batter_id] = {
                    'name': batter,
                    'bat_hand': bat_hand,
                    'batter_id': batter_id,
                    'total_runs': stats.runs,
                    'total_balls': stats.balls,
                    'dismissals': stats.dismissals,
                    'dot_balls': stats.dot_balls,
                    'singles': stats.singles,
                    'fours': stats.fours,
                    'sixes': stats.sixes,
                    'strike_rate': stats.strike_rate,
                    'average': stats.average,
                    'effective_strike_rate': effective_sr,
                    'effective_average': effective_avg,
                    'vulnerability': vulnerability,
                    'by_phase': phase_stats,
                    'vs_bowler_styles': bowl_style_stats,
                    'vs_batting_hand': batting_hand_stats,
                    'by_bat_pos': bat_pos_stats,
                    'vs_line_length': line_length_stats,
                    'phase_line_length': phase_ll_stats,
                    'style_line_length': style_ll_stats
                }
                
            except Exception as e:
                print(f"Error processing batter {batter}: {str(e)}")
                continue
        
        return profiles

    def _process_analysis(self, data: pd.DataFrame, analysis_type: str) -> Dict:
        """Process a specific type of analysis"""
        if analysis_type not in self.ANALYSIS_CONFIGS:
            return {}
            
        config = self.ANALYSIS_CONFIGS[analysis_type]
        groups = DataFrameProcessor.process_groups(data, config)
        
        results = {}
        for group_key, group_data in groups:
            # Skip unknown entries
            if not DataFrameProcessor.clean_group_key(group_key) or group_key == "unknown":
                continue
                
            # For line-length analysis, skip if either line or length is unknown
            if isinstance(group_key, tuple) and len(group_key) == 2:
                if "unknown" in group_key:
                    continue
                    
            stats = self.stats_processor.process_group(group_data, config.min_balls)
            if stats is None:
                continue
                
            # Handle line-length formatting
            if isinstance(group_key, tuple) and len(group_key) == 2 and 'line' in data.columns:
                line_display, length_display = DataFrameProcessor.format_line_length(group_key[0], group_key[1])
                # Skip if either line or length is unknown
                if "unknown" in (line_display, length_display):
                    continue
                results[(line_display, length_display)] = stats.__dict__
            else:
                # Convert numeric keys to integers if they are whole numbers
                if isinstance(group_key, (int, float)) and float(group_key).is_integer():
                    group_key = int(group_key)
                results[group_key] = stats.__dict__
                
        return results

    def analyze_batter(self, 
                      batter: str, 
                      analysis_type: str, 
                      filters: Optional[Dict] = None) -> Dict:
        """Analyze batter statistics with caching"""
        if self.data is not None:
            batter_data = self.data[self.data['bat'] == batter]
            return self._process_analysis(batter_data, analysis_type)
            
        # Get from saved profiles
        profile = self.batter_profiles.get(batter)
        if not profile:
            return None
            
        # Get the relevant section and apply filters
        section_map = {
            'overall': profile,
            'phase': 'by_phase',
            'bowling_style': 'vs_bowler_styles',
            'batting_hand': 'vs_batting_hand',
            'bat_pos': 'by_bat_pos',
            'line_length': 'vs_line_length',
            'phase_line_length': 'phase_line_length',
            'style_line_length': 'style_line_length'
        }
        
        section = profile.get(section_map.get(analysis_type, ''), {})
        if not filters:
            return section
            
        return {k: v for k, v in section.items() if self._matches_filters(k, v, filters)}
        
    def _matches_filters(self, key: Any, value: Dict, filters: Dict) -> bool:
        """Check if a section matches the given filters"""
        for filter_key, filter_value in filters.items():
            if filter_key == 'phase' and key != filter_value:
                return False
            elif filter_key == 'bowl_style' and key != filter_value:
                return False
            elif filter_key == 'bat_hand' and key != filter_value:
                return False
        return True

    def get_available_bowling_styles(self) -> List[str]:
        """Get list of available bowling styles"""
        if self.data is not None:
            return DataFrameProcessor.get_unique_values(self.data, 'bowl_style')
        return sorted(set(
            style for profile in self.batter_profiles.values() 
            for style in profile.get('vs_bowler_styles', {}).keys()
        ))

    def get_available_batting_hands(self) -> List[str]:
        """Get list of available batting hands"""
        if self.data is not None:
            return DataFrameProcessor.get_unique_values(self.data, 'bat_hand')
        return sorted(set(
            hand for profile in self.batter_profiles.values() 
            for hand in profile.get('vs_batting_hand', {}).keys()
        ))

    def get_all_batters(self) -> List[str]:
        """Get list of all batters"""
        return list(self.batter_profiles.keys())