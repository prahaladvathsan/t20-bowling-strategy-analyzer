from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import json
from pathlib import Path
from functools import lru_cache
from data_processor import DataFrameProcessor, GroupConfig
from bowler_processor import BowlerProcessor, BowlingStats

class BowlerAnalyzer:
    """Analyzes bowler effectiveness with different lines and lengths"""
    
    # Analysis configurations
    ANALYSIS_CONFIGS = {
        'overall': GroupConfig(group_by=None, min_balls=10),
        'phase': GroupConfig(group_by='phase', min_balls=10),
        'batting_hand': GroupConfig(group_by='bat_hand', min_balls=5),
        'line_length': GroupConfig(group_by=['line', 'length'], min_balls=1),
        'phase_line_length': GroupConfig(group_by=['phase', 'line', 'length'], min_balls=1)
    }
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """Initialize with ball-by-ball dataset or load from saved profiles"""
        if data is not None:
            self.data = data.copy()
            
            # Validate critical columns
            missing_cols = DataFrameProcessor.validate_columns(
                self.data, 
                ['bowl', 'bowlruns', 'ballfaced', 'out']
            )
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Initialize processors
            self.stats_processor = BowlerProcessor()
            
            # Pre-calculate profiles
            self.bowler_profiles = self._create_bowler_profiles()
        else:
            # Load from saved profiles
            self.bowler_profiles = self._load_profiles()
            self.data = None
    
    def _load_profiles(self) -> Dict:
        """Load bowler profiles from saved JSON file"""
        try:
            db_path = Path(__file__).parent.parent / "db"
            with open(db_path / "bowler_profiles.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise ValueError("No saved profiles found. Please run the backend processor first.")
    
    def _create_bowler_profiles(self) -> Dict:
        """Create profiles for bowlers based on their performance"""
        profiles = {}
        
        for bowler, bowler_data in self.data.groupby('bowl'):
            try:
                # Get basic stats
                stats = self.stats_processor.extract_bowling_stats(bowler_data)
                
                # Get bowler type
                bowl_kind = bowler_data['bowl_kind'].iloc[0] if 'bowl_kind' in bowler_data.columns else 'Unknown'
                bowl_style = bowler_data['bowl_style'].iloc[0] if 'bowl_style' in bowler_data.columns else 'Unknown'
                
                # Process different analysis types
                phase_stats = self._process_analysis(bowler_data, 'phase')
                batting_hand_stats = self._process_analysis(bowler_data, 'batting_hand')
                line_length_stats = self._process_analysis(bowler_data, 'line_length')
                phase_ll_stats = self._process_analysis(bowler_data, 'phase_line_length')
                
                # Create profile
                profiles[bowler] = {
                    'bowl_kind': bowl_kind,
                    'bowl_style': bowl_style,
                    'total_runs': stats.runs,
                    'total_balls': stats.balls,
                    'wickets': stats.wickets,
                    'dot_balls': stats.dot_balls,
                    'singles': stats.singles,
                    'fours': stats.fours,
                    'sixes': stats.sixes,
                    'no_balls': stats.no_balls,
                    'wides': stats.wides,
                    'leg_byes': stats.leg_byes,
                    'byes': stats.byes,
                    'economy': stats.economy,
                    'average': stats.average,
                    'bowling_strike_rate': stats.bowling_strike_rate,
                    'by_phase': phase_stats,
                    'vs_batting_hand': batting_hand_stats,
                    'by_line_length': line_length_stats,
                    'phase_line_length': phase_ll_stats
                }
                
            except Exception as e:
                print(f"Error processing bowler {bowler}: {str(e)}")
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
            if not DataFrameProcessor.clean_group_key(group_key):
                continue
                
            stats = self.stats_processor.process_group(group_data, config.min_balls)
            if stats is None:
                continue
                
            # Handle line-length formatting
            if isinstance(group_key, tuple) and len(group_key) == 2 and 'line' in data.columns:
                line_display, length_display = DataFrameProcessor.format_line_length(group_key[0], group_key[1])
                results[(line_display, length_display)] = stats.__dict__
            else:
                results[group_key] = stats.__dict__
                
        return results
    
    def get_optimal_line_length(self, bowler: str) -> List[Dict]:
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
            combo_str = str(combo_key).strip("()")
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
                'strike_rate': stats['bowling_strike_rate'],
                'sample_size': stats['balls']
            })
        
        # Sort by effectiveness (lower is better)
        recommendations.sort(key=lambda x: x['effectiveness'])
        return recommendations[:3]  # Return top 3 recommendations
    
    def analyze_bowler(self, 
                      bowler: str, 
                      analysis_type: str, 
                      filters: Optional[Dict] = None) -> Dict:
        """Analyze bowler statistics with caching"""
        if self.data is not None:
            bowler_data = self.data[self.data['bowl'] == bowler]
            return self._process_analysis(bowler_data, analysis_type)
            
        # Get from saved profiles
        profile = self.bowler_profiles.get(bowler)
        if not profile:
            return None
            
        # Get the relevant section and apply filters
        section_map = {
            'overall': profile,
            'phase': 'by_phase',
            'batting_hand': 'vs_batting_hand',
            'line_length': 'by_line_length',
            'phase_line_length': 'phase_line_length'
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
            elif filter_key == 'bat_hand' and key != filter_value:
                return False
        return True
    
    def get_bowler_profile(self, bowler: str) -> Optional[Dict]:
        """Return profile for a specific bowler"""
        return self.bowler_profiles.get(bowler)
    
    def get_available_batting_hands(self) -> List[str]:
        """Get list of available batting hands"""
        if self.data is not None:
            return DataFrameProcessor.get_unique_values(self.data, 'bat_hand')
        return sorted(set(
            hand for profile in self.bowler_profiles.values() 
            for hand in profile.get('vs_batting_hand', {}).keys()
        ))
    
    def get_all_bowlers(self) -> List[str]:
        """Return a list of all bowlers in the profiles"""
        return list(self.bowler_profiles.keys())