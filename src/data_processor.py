import pandas as pd
import numpy as np

class DataProcessor:
    """
    Class for cleaning and preprocessing raw ball-by-ball T20 cricket data.
    """
    
    # Add class-level display mappings that can be accessed by other modules
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
    
    def process(self):
        """
        Perform data cleaning and preprocessing.
        
        Returns:
        --------
        pandas.DataFrame
            Processed DataFrame with standardized columns and filled missing values
        """
        try:
            # Make a copy to avoid modifying the original
            df = self.raw_data.copy()
            
            # Ensure required columns exist
            self._validate_required_columns(df)
            
            # Fill missing values with appropriate defaults
            self._fill_missing_values(df)
            
            # Standardize column values
            self._standardize_columns(df)
            
            # Create derived columns if needed
            self._create_derived_columns(df)
            
            # Store the processed data
            self.processed_data = df
            
            return self.processed_data
            
        except Exception as e:
            print(f"Error in data processing: {str(e)}")
            raise
    
    def _create_derived_columns(self, df):
        # Check if wagon coordinates exist
        if 'wagon_x' in df.columns and 'wagon_y' in df.columns:
            # Calculate shot distance from origin (180,180)
            df['shot_distance'] = np.sqrt(
                (df['wagon_x'] - 180) ** 2 + 
                (df['wagon_y'] - 180) ** 2
            )
            
            # Calculate shot angle (in degrees) from origin
            # Using atan2 to get angle in correct quadrant
            df['shot_angle'] = np.degrees(
                np.arctan2(
                    df['wagon_y'] - 180,
                    df['wagon_x'] - 180
                )
            )
    
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
            LINE_MAPPING = {
                'WIDE_OUTSIDE_OFFSTUMP': 0,
                'OUTSIDE_OFFSTUMP': 1, 
                'ON_THE_STUMPS': 2, 
                'DOWN_LEG': 3, 
                'WIDE_DOWN_LEG': 4
            }
            
            LINE_DISPLAY = {
                0: 'Wide Outside Off',
                1: 'Outside Off',
                2: 'On Stumps',
                3: 'Down Leg',
                4: 'Wide Down Leg'
            }
            
            # Convert all values to uppercase strings first
            df['line'] = df['line'].astype(str).str.upper()
            
            # Store original values in _code columns
            df['line_code'] = df['line']
            
            # First map to numeric values
            df['line'] = df['line'].map(lambda x: LINE_MAPPING.get(x, 2))  # Default to 2 (ON_THE_STUMPS) for unknown
            
            # Create display column for visualization
            df['line_display'] = df['line'].map(lambda x: LINE_DISPLAY.get(x, 'On Stumps'))
        
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
            
            LENGTH_DISPLAY = {
                0: 'Full Toss',
                1: 'Yorker',
                2: 'Full',
                3: 'Good Length',
                4: 'Short Good Length',
                5: 'Short'
            }
            
            # Convert all values to uppercase strings first
            df['length'] = df['length'].astype(str).str.upper()
            
            # Store original values in _code columns
            df['length_code'] = df['length']
            
            # First map to numeric values
            df['length'] = df['length'].map(lambda x: LENGTH_MAPPING.get(x, 3))  # Default to 3 (GOOD_LENGTH) for unknown
            
            # Create display column for visualization
            df['length_display'] = df['length'].map(lambda x: LENGTH_DISPLAY.get(x, 'Good Length'))
    
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