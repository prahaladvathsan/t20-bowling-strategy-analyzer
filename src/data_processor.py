"""Data processing utilities for cricket analytics"""
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class GroupConfig:
    """Configuration for data grouping"""
    group_by: Optional[str | List[str]]
    min_balls: int = 5
    filters: Optional[Dict[str, Any]] = None

class DataFrameProcessor:
    """Handles common DataFrame operations for cricket analysis"""
    
    # Display mappings
    LINE_DISPLAY = {
        1: "Wide Outside Off",
        2: "Outside Off",
        3: "Off",
        4: "Middle",
        5: "Leg",
        6: "Wide Outside Leg"
    }
    
    LENGTH_DISPLAY = {
        1: "Yorker",
        2: "Full",
        3: "Good",
        4: "Short",
        5: "Bouncer"
    }
    
    @staticmethod
    def apply_filters(data: pd.DataFrame, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Apply filters to DataFrame"""
        if not filters:
            return data
            
        filtered_data = data.copy()
        for col, value in filters.items():
            if col in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[col] == value]
        return filtered_data
    
    @staticmethod
    def validate_columns(data: pd.DataFrame, required_cols: List[str]) -> List[str]:
        """Validate required columns exist in DataFrame"""
        return [col for col in required_cols if col not in data.columns]
    
    @staticmethod
    def clean_group_key(key: Any) -> bool:
        """Check if group key is valid"""
        if isinstance(key, tuple):
            return not any(pd.isna(k) or k == '-' or k == '' for k in key)
        return not (pd.isna(key) or key == '-' or key == '')
    
    @staticmethod
    def format_line_length(line: int, length: int) -> Tuple[str, str]:
        """Convert line/length codes to display names"""
        line_display = DataFrameProcessor.LINE_DISPLAY.get(int(line), 'Unknown')
        length_display = DataFrameProcessor.LENGTH_DISPLAY.get(int(length), 'Unknown')
        return line_display, length_display
    
    @staticmethod
    def get_unique_values(data: pd.DataFrame, column: str) -> List[str]:
        """Get unique valid values from a column"""
        if column not in data.columns:
            return []
        values = data[column].dropna().unique()
        return sorted([str(v) for v in values if str(v) != '-' and str(v) != ''])
    
    @staticmethod
    def process_groups(data: pd.DataFrame, config: GroupConfig) -> Dict:
        """Process data according to grouping configuration"""
        if not config.group_by:
            return {}
            
        # Apply filters if any
        if config.filters:
            data = DataFrameProcessor.apply_filters(data, config.filters)
            
        # Handle missing columns
        if isinstance(config.group_by, list):
            if not all(col in data.columns for col in config.group_by):
                return {}
        elif config.group_by not in data.columns:
            return {}
            
        return data.groupby(config.group_by)

class DataProcessor:
    """
    Class for cleaning and preprocessing raw ball-by-ball T20 cricket data.
    """
    
    # Centralized display mappings
    LINE_DISPLAY = {
        0: 'Wide Outside Off',
        1: 'Outside Off',
        2: 'On Stumps',
        3: 'Down Leg',
        4: 'Wide Down Leg'
    }
    
    LENGTH_DISPLAY = {
        0: 'Full Toss',
        1: 'Yorker',
        2: 'Full',
        3: 'Good Length',
        4: 'Short Good Length',
        5: 'Short'
    }
    
    # Phase definitions
    PHASE_NAMES = {
        1: 'Powerplay (1-6)',
        2: 'Early Middle (7-12)',
        3: 'Late Middle (13-16)',
        4: 'Death (17-20)'
    }

    BATTING_POSITION_DISPLAY = {
        1: 'Opener',
        2: 'Opener',
        3: 'Top Order',
        4: 'Middle Order',
        5: 'Middle Order',
        6: 'Middle Order',
        7: 'Lower Order',
        8: 'Lower Order',
        9: 'Lower Order',
        10: 'Lower Order',
        11: 'Lower Order'
    }
    
    # Line and Length standardization mappings
    LINE_MAPPING = {
        'WIDE_OUTSIDE_OFFSTUMP': 0,
        'OUTSIDE_OFFSTUMP': 1, 
        'ON_THE_STUMPS': 2, 
        'DOWN_LEG': 3, 
        'WIDE_DOWN_LEG': 4
    }
    
    LENGTH_MAPPING = {
        'FULL_TOSS': 0,
        'YORKER': 1,
        'FULL': 2,
        'GOOD_LENGTH': 3,
        'SHORT_OF_A_GOOD_LENGTH': 4,
        'SHORT': 5
    }

    @staticmethod
    def calculate_strike_rate(runs, balls):
        """Calculate batting strike rate"""
        return (runs / balls * 100) if balls > 0 else 0
    
    @staticmethod
    def calculate_average(runs, dismissals):
        """Calculate batting/bowling average"""
        return (runs / dismissals) if dismissals > 0 else float('inf')
    
    @staticmethod
    def calculate_economy(runs, balls):
        """Calculate bowling economy rate"""
        return (runs / balls * 6) if balls > 0 else 0
    
    @staticmethod
    def calculate_bowling_strike_rate(balls, wickets):
        """Calculate bowling strike rate"""
        return (balls / wickets) if wickets > 0 else float('inf')
    
    def __init__(self, data):
        """
        Initialize the DataProcessor with raw data.
        
        Parameters:
        -----------
        data : pandas.DataFrame or str
            Raw ball-by-ball cricket data as a DataFrame or a file path to CSV
        """
        if isinstance(data, str):
            self.raw_data = pd.read_csv(data, low_memory=False)
        else:
            self.raw_data = data.copy()
        self.processed_data = None
    
    def _is_preprocessed(self, df):
        """
        Check if the data has already been preprocessed
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to check
            
        Returns:
        --------
        bool
            True if data appears to be preprocessed, False otherwise
        """
        # Check for processed data markers
        preprocessed_markers = [
            'line_display',
            'length_display',
            'shot_distance',
            'shot_angle',
            'p_bat_ns',
            'line_code',
            'length_code'
        ]
        
        # Check if critical derived columns exist
        has_derived_columns = all(col in df.columns for col in preprocessed_markers)
        
        # Check if line and length are already standardized (numeric)
        if 'line' in df.columns and 'length' in df.columns:
            line_standardized = pd.api.types.is_numeric_dtype(df['line'])
            length_standardized = pd.api.types.is_numeric_dtype(df['length'])
        else:
            line_standardized = False
            length_standardized = False
            
        return has_derived_columns and line_standardized and length_standardized

    def process(self):
        """
        Perform data cleaning and preprocessing if needed.
        
        Returns:
        --------
        pandas.DataFrame
            Processed DataFrame with standardized columns and filled missing values
        """
        try:
            # Make a copy to avoid modifying the original
            df = self.raw_data.copy()
            
            # Check if data is already preprocessed
            if self._is_preprocessed(df):
                print("Data appears to be preprocessed, skipping processing steps.")
                self._remove_missing_values(df)
                self.processed_data = df
                return self.processed_data
                
            # If not preprocessed, continue with processing
            print("Processing raw data...")
            
            # Ensure required columns exist
            self._validate_required_columns(df)
            
            # Fill missing values with appropriate defaults
            self._remove_missing_values(df)
            
            # Standardize column values
            self._standardize_columns(df)
            
            # Create derived columns if needed
            self._create_derived_columns(df)
            
            # Add non-striker batter ID column
            self._add_nonstriker_id(df)
            
            # Store the processed data
            self.processed_data = df
            
            return self.processed_data
            
        except Exception as e:
            print(f"Error in data processing: {str(e)}")
            raise
    
    def _add_nonstriker_id(self, df):
        """
        Add a column for the non-striker batter ID.
        
        For each ball in the match, determine who the non-striker is by tracking
        both batters at the crease and updating after dismissals.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to process
        """
        # Check if required columns exist
        required_cols = ['p_match', 'inns', 'p_bat', 'out', 'p_out']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: Cannot add non-striker ID due to missing required columns: {[col for col in required_cols if col not in df.columns]}")
            return
            
        # Initialize the non-striker column
        df['p_bat_ns'] = np.nan
        
        # Process each match and innings separately
        for (match_id, inns_id), innings_group in df.groupby(['p_match', 'inns']):
            # Sort by ball_id or ball to ensure chronological order
            if 'ball_id' in innings_group.columns:
                innings_data = innings_group.sort_values('ball_id')
            elif 'ball' in innings_group.columns:
                innings_data = innings_group.sort_values('ball')
            else:
                innings_data = innings_group
            
            # Get the indices for this innings
            innings_indices = innings_data.index.tolist()
            
            # Track batters at the crease [striker_id, non_striker_id]
            batters_at_crease = [None, None]
            
            # Keep track of the start index where non-striker is None
            ns_unknown_start_idx = 0
            
            # Process each ball in order
            for i, idx in enumerate(innings_indices):
                row = df.loc[idx]
                
                # Get current striker
                current_striker_id = row['p_bat']
                
                # Handle first ball of innings
                if i == 0:
                    batters_at_crease[0] = current_striker_id
                    # Non-striker will be discovered when they come on strike
                    continue
                
                # Check if a wicket fell in the previous ball
                prev_row = df.loc[innings_indices[i-1]]
                if prev_row['out'] and pd.notna(prev_row['p_out']):
                    dismissed_id = prev_row['p_out']
                    
                    # Update batters_at_crease
                    if dismissed_id == batters_at_crease[0]:
                        # Striker was dismissed, new batsman comes in
                        batters_at_crease[0] = current_striker_id
                    elif dismissed_id == batters_at_crease[1]:
                        # Non-striker was dismissed (run out), new batsman comes in
                        # But we don't know who they are yet until they face a ball
                        batters_at_crease[1] = None
                        ns_unknown_start_idx = i  # Start tracking from this point
                
                # If current striker is different from previous ball's striker
                if current_striker_id != batters_at_crease[0]:
                    # Must be the non-striker coming on strike or a new batsman
                    if batters_at_crease[1] is None:
                        # First time seeing this non-striker - update all previous unknown non-strikers
                        newly_identified_nonstriker = batters_at_crease[0]
                        
                        # Fill in all previous balls since ns_unknown_start_idx
                        for j in range(ns_unknown_start_idx, i):
                            prev_idx = innings_indices[j]
                            df.at[prev_idx, 'p_bat_ns'] = current_striker_id
                        
                        # Update current batters at crease
                        batters_at_crease = [current_striker_id, newly_identified_nonstriker]
                    else:
                        # Regular rotation of strike
                        batters_at_crease = [current_striker_id, batters_at_crease[0]]
                
                # Set the non-striker ID for current ball
                df.at[idx, 'p_bat_ns'] = batters_at_crease[1]
        
        return df
    
    def _create_derived_columns(self, df):
        # Check if wagon coordinates exist
        if 'wagonX' in df.columns and 'wagonY' in df.columns:
            # Calculate shot distance from origin (180,180)
            df['shot_distance'] = np.sqrt(
                (df['wagonX'] - 180) ** 2 + 
                (df['wagonY'] - 180) ** 2
            )
            
            # Calculate shot angle (in degrees) from origin
            # Using atan2 to get angle in correct quadrant
            df['shot_angle'] = np.degrees(
                np.arctan2(
                    df['wagonY'] - 180,
                    df['wagonX'] - 180
                )
            )
    
    def _remove_missing_values(self, df):
        """
        Remove missing values from the DataFrame.
        """
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            df[col] = df[col].dropna()
        
        # Fill missing numeric columns with appropriate defaults
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # Specific handling for some important numeric columns
        if 'score' in numeric_cols:
            df['score'] = df['score'].fillna(0)
        
        if 'out' in df.columns:
            df['out'] = df['out'].fillna(False)
        
        
        # Fill remaining numeric columns with 0
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(0)
    
    def _fill_missing_values(self, df):
        """
        Fill missing values with appropriate defaults.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to process
        """
        # Fill missing string columns with 'Unknown'
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            df[col] = df[col].fillna('Unknown')
        
        # Fill missing numeric columns with appropriate defaults
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # Specific handling for some important numeric columns
        if 'score' in numeric_cols:
            df['score'] = df['score'].fillna(0)
        
        if 'out' in df.columns:
            df['out'] = df['out'].fillna(False)
        
        
        # Fill remaining numeric columns with 0
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(0)
    
    def _standardize_columns(self, df):
        """
        Standardize column values for consistency.
        """
        # Standardize bowling line
        if 'line' in df.columns:
            # Convert all values to uppercase strings first
            df['line'] = df['line'].astype(str).str.upper()
            
            # Store original values in _code columns
            df['line_code'] = df['line']
            
            # First map to numeric values
            df['line'] = df['line_code'].map(lambda x: self.LINE_MAPPING.get(x, 5))  # Default to 2 (ON_THE_STUMPS) for unknown
            
            # Create display column for visualization
            df['line_display'] = df['line'].map(lambda x: self.LINE_DISPLAY.get(x, 'On Stumps'))
        
        # Standardize bowling length
        if 'length' in df.columns:
            # Convert all values to uppercase strings first
            df['length'] = df['length'].astype(str).str.upper()
            
            # Store original values in _code columns
            df['length_code'] = df['length']
            
            # First map to numeric values
            df['length'] = df['length_code'].map(lambda x: self.LENGTH_MAPPING.get(x, 6))  # Default to 3 (GOOD_LENGTH) for unknown
            
            # Create display column for visualization
            df['length_display'] = df['length'].map(lambda x: self.LENGTH_DISPLAY.get(x, 'Good Length'))
    
    def _validate_required_columns(self, df):
        """
        Validate that all required columns exist in the DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to validate
        """
        required_columns = ['p_match', 'bat', 'bowl', 'line', 'length', 'phase', 'score', 'out']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    def get_processed_data(self):
        """
        Get the processed data.
        
        Returns:
        --------
        pandas.DataFrame
            Processed DataFrame
        """
        if self.processed_data is None:
            raise ValueError("Data has not been processed yet. Call process() first.")
        return self.processed_data


# Example usage
if __name__ == "__main__":
    # Example with file path
    # processor = DataProcessor("data/sample_t20_data.csv")
    
    # Example with DataFrame
    # import pandas as pd
    # raw_data = pd.read_csv("data/sample_t20_data.csv")
    # processor = DataProcessor(raw_data)
    
    # Process the data
    # processed_data = processor.process()
    # print(processed_data.head())
    pass