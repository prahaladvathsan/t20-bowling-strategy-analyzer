import pandas as pd
import numpy as np

class DataProcessor:
    """
    Class for cleaning and preprocessing raw ball-by-ball T20 cricket data.
    """
    
    def __init__(self, data):
        """
        Initialize the DataProcessor with raw data.
        
        Parameters:
        -----------
        data : pandas.DataFrame or str
            Raw ball-by-ball cricket data as a DataFrame or a file path to CSV
        """
        if isinstance(data, str):
            self.raw_data = pd.read_csv(data)
        else:
            self.raw_data = data.copy()
        self.processed_data = None
    
    def process(self):
        """
        Perform data cleaning and preprocessing.
        
        Returns:
        --------
        pandas.DataFrame
            Processed DataFrame with standardized columns and filled missing values
        """
        # Make a copy to avoid modifying the original
        df = self.raw_data.copy()
        
        # Fill missing values with appropriate defaults
        self._fill_missing_values(df)
        
        # Standardize column values
        self._standardize_columns(df)
        
        # Create derived columns if needed
        self._create_derived_columns(df)
        
        # Ensure required columns exist
        self._validate_required_columns(df)
        
        # Store the processed data
        self.processed_data = df
        
        return self.processed_data
    
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
        
        if 'phase' in numeric_cols:
            # Determine phase from over number if missing
            mask = df['phase'].isna()
            if mask.any():
                # T20 has specific phase definitions (as provided):
                # Phase 1: Overs 1-6 (Powerplay)
                # Phase 2: Overs 7-12 (Middle overs first part)
                # Phase 3: Overs 13-16 (Middle overs second part)
                # Phase 4: Overs 17-20 (Death overs)
                
                # Assign phases based on over number
                conditions = [
                    (df['over'] >= 1) & (df['over'] <= 6),
                    (df['over'] >= 7) & (df['over'] <= 12),
                    (df['over'] >= 13) & (df['over'] <= 16),
                    (df['over'] >= 17) & (df['over'] <= 20)
                ]
                choices = [1, 2, 3, 4]
                
                df.loc[mask, 'phase'] = np.select(conditions, choices, default=0)
        
        # Fill remaining numeric columns with 0
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(0)
    
    def _standardize_columns(self, df):
        """
        Standardize column values for consistency.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to process
        """
        # Standardize phase values
        if 'phase' in df.columns:
            # Ensure phase is an integer between 1-4
            df['phase'] = df['phase'].astype(float).astype('Int64')
            # Validate and correct phase values
            mask = ~df['phase'].isin([1, 2, 3, 4])
            if mask.any():
                # For invalid phases, determine from over number
                conditions = [
                    (df['over'] >= 1) & (df['over'] <= 6),
                    (df['over'] >= 7) & (df['over'] <= 12),
                    (df['over'] >= 13) & (df['over'] <= 16),
                    (df['over'] >= 17) & (df['over'] <= 20)
                ]
                choices = [1, 2, 3, 4]
                df.loc[mask, 'phase'] = np.select(conditions, choices, default=0)
        
        # Standardize bowling line
        if 'line' in df.columns:
            LINE_MAPPING = {
                'WIDE_OUTSIDE_OFFSTUMP': 0,
                'OUTSIDE_OFFSTUMP': 1, 
                'ON_THE_STUMPS': 2, 
                'DOWN_LEG': 3, 
                'WIDE_DOWN_LEG': 4
            }
            # Create a reverse mapping to standardize values
            reverse_line_mapping = {v: k for k, v in LINE_MAPPING.items()}
            
            # If the value is already a number, use reverse mapping
            # Otherwise, convert to uppercase and use as is
            df['line_code'] = df['line'].apply(
                lambda x: x if pd.isna(x) else (
                    reverse_line_mapping.get(x, x) if isinstance(x, (int, float)) else str(x).upper()
                )
            )
        
        # Standardize bowling length
        if 'length' in df.columns:
            LENGTH_MAPPING = {
                'FULL_TOSS': 0, 
                'YORKER': 1, 
                'FULL': 2, 
                'GOOD_LENGTH': 3, 
                'SHORT_OF_A_GOOD_LENGTH': 4, 
                'SHORT': 5
            }
            # Create a reverse mapping to standardize values
            reverse_length_mapping = {v: k for k, v in LENGTH_MAPPING.items()}
            
            # If the value is already a number, use reverse mapping
            # Otherwise, convert to uppercase and use as is
            df['length_code'] = df['length'].apply(
                lambda x: x if pd.isna(x) else (
                    reverse_length_mapping.get(x, x) if isinstance(x, (int, float)) else str(x).upper()
                )
            )
        
        # Identify bowl_kind from bowl_style if missing
        if 'bowl_style' in df.columns and 'bowl_kind' in df.columns:
            # Create a mapping of bowl_style prefixes to determine bowl_kind
            pace_prefixes = ['RF', 'RFM', 'RMF', 'LF', 'LFM', 'LMF', 'RM', 'LM', 'RS', 'LS']
            spin_prefixes = ['LB', 'LWS', 'SLA', 'OB', 'LBG', 'RAB', 'LAB']
            
            # Fill missing bowl_kind values based on bowl_style
            mask = df['bowl_kind'].isna() | (df['bowl_kind'] == 'Unknown')
            if mask.any():
                # Function to determine bowl_kind from bowl_style
                def get_bowl_kind(style):
                    if pd.isna(style) or style == '-' or style == 'Unknown':
                        return 'Unknown'
                    
                    style_parts = style.split('/')
                    for part in style_parts:
                        part = part.strip()
                        for prefix in pace_prefixes:
                            if part.startswith(prefix):
                                return 'pace bowler'
                        for prefix in spin_prefixes:
                            if part.startswith(prefix):
                                return 'spin bowler'
                    
                    return 'Unknown'
                
                df.loc[mask, 'bowl_kind'] = df.loc[mask, 'bowl_style'].apply(get_bowl_kind)
    
    def _create_derived_columns(self, df):
        """
        Create useful derived columns for analysis.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to process
        """
        # Process wagon wheel data
        # Wagon wheel coordinates have (180,180) as origin with radius 180
        # Pitch length is 30 units (from 165 to 195 on y scale)
        if all(col in df.columns for col in ['wagonX', 'wagonY']):
            # Calculate distance from origin (center of the field)
            df['shot_distance'] = np.sqrt((df['wagonX'] - 180)**2 + (df['wagonY'] - 180)**2)
            
            # Calculate shot angle (in degrees, 0 is straight, positive is offside for RHB)
            df['shot_angle'] = np.degrees(np.arctan2(df['wagonX'] - 180, df['wagonY'] - 180))
    
    def _validate_required_columns(self, df):
        """
        Ensure that required columns exist in the DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to validate
        
        Raises:
        -------
        ValueError
            If required columns are missing
        """
        required_columns = ['bat', 'bowl', 'score', 'out']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Required columns missing: {', '.join(missing_columns)}")
        
        # Check if important columns for better analysis exist, add them if not
        recommended_columns = ['line', 'length', 'phase', 'bat_hand', 'bowl_style', 'bowl_kind']
        for col in recommended_columns:
            if col not in df.columns:
                print(f"Warning: Recommended column '{col}' is missing. Analysis may be limited.")

    def get_processed_data(self):
        """
        Return the processed data or process it if not done yet.
        
        Returns:
        --------
        pandas.DataFrame
            Processed DataFrame
        """
        if self.processed_data is None:
            return self.process()
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