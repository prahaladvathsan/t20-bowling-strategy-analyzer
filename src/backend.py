import pandas as pd
import json
import os
import numpy as np
from pathlib import Path
from data_processor import DataProcessor
from batter_analyzer import BatterVulnerabilityAnalyzer
from bowler_analyzer import BowlerAnalyzer
from bowling_plan_generator import BowlingPlanGenerator

class Backend:
    """Backend class for processing and storing cricket analysis data"""
    
    def __init__(self, data_path):
        """Initialize backend with data path"""
        self.data_path = Path(data_path)
        self.db_path = Path(__file__).parent.parent / "db"
        self.db_path.mkdir(exist_ok=True)
        
    def process_and_store_data(self):
        """Process data and store analysis results"""
        print("Loading and processing data...")
        data = pd.read_csv(self.data_path, low_memory=False)
        
        # Process data
        processor = DataProcessor(data)
        processed_data = processor.process()
        
        # Initialize analyzers
        print("Initializing analyzers...")
        batter_analyzer = BatterVulnerabilityAnalyzer(processed_data)
        bowler_analyzer = BowlerAnalyzer(processed_data)
        plan_generator = BowlingPlanGenerator(processed_data)
        
        # Save processed data and analysis results
        print("Saving processed data and analysis results...")
        self._save_processed_data(processed_data)
        self._save_batter_profiles(batter_analyzer.batter_profiles)
        self._save_bowler_profiles(bowler_analyzer.bowler_profiles)
        self._save_plan_generator_data(plan_generator)
        
        print("Data processing and storage complete!")
        
    def _save_processed_data(self, data):
        """Save processed DataFrame"""
        data.to_parquet(self.db_path / "processed_data.parquet")
        
    def _save_batter_profiles(self, profiles):
        """Save batter profiles"""
        # Convert numpy types to Python native types for JSON serialization
        serializable_profiles = self._make_json_serializable(profiles)
        with open(self.db_path / "batter_profiles.json", "w") as f:
            json.dump(serializable_profiles, f)
            
    def _save_bowler_profiles(self, profiles):
        """Save bowler profiles"""
        serializable_profiles = self._make_json_serializable(profiles)
        with open(self.db_path / "bowler_profiles.json", "w") as f:
            json.dump(serializable_profiles, f)
            
    def _save_plan_generator_data(self, plan_generator):
        """Save plan generator data"""
        data = {
            'batter_profiles': self._make_json_serializable(plan_generator.batter_profiles),
            'bowler_profiles': self._make_json_serializable(plan_generator.bowler_profiles),
            'phase_insights': self._make_json_serializable(plan_generator.phase_insights)
        }
        with open(self.db_path / "plan_generator_data.json", "w") as f:
            json.dump(data, f)
            
    def _make_json_serializable(self, obj):
        """Convert object to JSON serializable format"""
        if isinstance(obj, dict):
            return {str(k): self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(i) for i in obj]
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
            
if __name__ == "__main__":
    # Path to your data file
    data_file = Path(__file__).parent.parent / "data" / "t20_bbb.csv"
    
    # Initialize and run backend
    backend = Backend(data_file)
    backend.process_and_store_data()