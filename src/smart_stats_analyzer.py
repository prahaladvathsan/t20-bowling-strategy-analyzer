import pandas as pd
import numpy as np
import json
from pathlib import Path

class SmartStatsAnalyzer:
    """
    Analyzes cricket data using ESPNcricinfo Smart Stats metrics
    """
    
    def __init__(self, smart_metrics=None):
        """Initialize with smart metrics data or load from saved profiles"""
        if smart_metrics is not None:
            self.smart_metrics = smart_metrics
        else:
            # Load from saved profiles
            self.smart_metrics = self._load_metrics()
            
        # Initialize useful lookups
        self.player_id_to_name = {}
        self._initialize_player_lookups()
    
    def _load_metrics(self):
        """Load smart metrics from saved JSON file"""
        try:
            db_path = Path(__file__).parent.parent / "db"
            with open(db_path / "smart_metrics.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise ValueError("No saved smart metrics found. Please run the backend processor first.")
    
    def _load_player_qualities(self):
        """Load player quality indices"""
        try:
            db_path = Path(__file__).parent.parent / "db"
            with open(db_path / "player_qualities.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _initialize_player_lookups(self):
        """Initialize player ID to name lookups from smart metrics"""
        player_qualities = self._load_player_qualities()
        
        # Build ID to name map from player qualities
        for player_id, qualities in player_qualities.items():
            if 'player_name' in qualities:
                self.player_id_to_name[player_id] = qualities['player_name']
    
    def get_all_matches(self):
        """Return a list of all matches in the smart metrics"""
        return list(self.smart_metrics.keys())
    
    def get_smart_runs_ranking(self, match_id=None):
        """
        Get batsmen ranked by Smart Runs
        
        Parameters:
        -----------
        match_id : float or None
            Match ID to analyze (None for all matches)
        
        Returns:
        --------
        list
            List of batsmen with their Smart Runs, sorted in descending order
        """
        # If specific match requested
        if match_id is not None and str(match_id) in self.smart_metrics:
            match_data = self.smart_metrics[str(match_id)]
            smart_runs = match_data.get('smart_runs', {})
            
            # Format the rankings
            rankings = []
            for batsman, stats in smart_runs.items():
                rankings.append({
                    'player': batsman,
                    'player_id': stats.get('player_id'),
                    'smart_runs': stats.get('smart_runs', 0),
                    'conventional_runs': stats.get('conventional_runs', 0),
                    'balls_faced': stats.get('balls_faced', 0),
                    'smart_sr': stats.get('smart_sr', 0),
                    'conventional_sr': stats.get('conventional_sr', 0)
                })
            
            # Sort by smart runs
            rankings.sort(key=lambda x: x['smart_runs'], reverse=True)
            return rankings
        
        # Aggregate across all matches
        aggregated_runs = {}
        
        # Process each match
        for match_id, match_data in self.smart_metrics.items():
            smart_runs = match_data.get('smart_runs', {})
            
            for batsman, stats in smart_runs.items():
                if batsman not in aggregated_runs:
                    aggregated_runs[batsman] = {
                        'player': batsman,
                        'player_id': stats.get('player_id'),
                        'smart_runs': 0,
                        'conventional_runs': 0,
                        'balls_faced': 0,
                        'matches': 0
                    }
                
                # Aggregate stats
                aggregated_runs[batsman]['smart_runs'] += stats.get('smart_runs', 0)
                aggregated_runs[batsman]['conventional_runs'] += stats.get('conventional_runs', 0)
                aggregated_runs[batsman]['balls_faced'] += stats.get('balls_faced', 0)
                aggregated_runs[batsman]['matches'] += 1
        
        # Calculate strike rates
        for player, stats in aggregated_runs.items():
            if stats['balls_faced'] > 0:
                stats['smart_sr'] = (stats['smart_runs'] / stats['balls_faced']) * 100
                stats['conventional_sr'] = (stats['conventional_runs'] / stats['balls_faced']) * 100
            else:
                stats['smart_sr'] = 0
                stats['conventional_sr'] = 0
        
        # Convert to list and sort
        rankings = list(aggregated_runs.values())
        rankings.sort(key=lambda x: x['smart_runs'], reverse=True)
        
        return rankings
    
    def get_smart_wickets_ranking(self, match_id=None):
        """
        Get bowlers ranked by Smart Wickets
        
        Parameters:
        -----------
        match_id : float or None
            Match ID to analyze (None for all matches)
        
        Returns:
        --------
        list
            List of bowlers with their Smart Wickets, sorted in descending order
        """
        # If specific match requested
        if match_id is not None and str(match_id) in self.smart_metrics:
            match_data = self.smart_metrics[str(match_id)]
            smart_wickets = match_data.get('smart_wickets', {})
            
            # Format the rankings
            rankings = []
            for bowler, stats in smart_wickets.items():
                rankings.append({
                    'player': bowler,
                    'player_id': stats.get('player_id'),
                    'smart_wickets': stats.get('smart_wickets', 0),
                    'conventional_wickets': stats.get('conventional_wickets', 0),
                    'total_overs': stats.get('total_overs', 0),
                    'total_runs': stats.get('total_runs', 0),
                    'smart_er': stats.get('smart_er', 0),
                    'conventional_er': stats.get('conventional_er', 0)
                })
            
            # Sort by smart wickets
            rankings.sort(key=lambda x: x['smart_wickets'], reverse=True)
            return rankings
        
        # Aggregate across all matches
        aggregated_wickets = {}
        
        # Process each match
        for match_id, match_data in self.smart_metrics.items():
            smart_wickets = match_data.get('smart_wickets', {})
            
            for bowler, stats in smart_wickets.items():
                if bowler not in aggregated_wickets:
                    aggregated_wickets[bowler] = {
                        'player': bowler,
                        'player_id': stats.get('player_id'),
                        'smart_wickets': 0,
                        'conventional_wickets': 0,
                        'total_overs': 0,
                        'total_runs': 0,
                        'matches': 0
                    }
                
                # Aggregate stats
                aggregated_wickets[bowler]['smart_wickets'] += stats.get('smart_wickets', 0)
                aggregated_wickets[bowler]['conventional_wickets'] += stats.get('conventional_wickets', 0)
                aggregated_wickets[bowler]['total_overs'] += stats.get('total_overs', 0)
                aggregated_wickets[bowler]['total_runs'] += stats.get('total_runs', 0)
                aggregated_wickets[bowler]['matches'] += 1
        
        # Calculate economy rates
        for player, stats in aggregated_wickets.items():
            if stats['total_overs'] > 0:
                stats['conventional_er'] = stats['total_runs'] / stats['total_overs']
                
                # Smart economy rate
                if stats['conventional_wickets'] > 0:
                    smart_ratio = stats['smart_wickets'] / stats['conventional_wickets']
                    smart_runs_conceded = stats['total_runs'] / smart_ratio
                    stats['smart_er'] = smart_runs_conceded / stats['total_overs']
                else:
                    stats['smart_er'] = stats['conventional_er']
            else:
                stats['conventional_er'] = 0
                stats['smart_er'] = 0
        
        # Convert to list and sort
        rankings = list(aggregated_wickets.values())
        rankings.sort(key=lambda x: x['smart_wickets'], reverse=True)
        
        return rankings
    
    def get_player_impact_ranking(self, match_id=None):
        """
        Get players ranked by Impact Score
        
        Parameters:
        -----------
        match_id : float or None
            Match ID to analyze (None for all matches)
        
        Returns:
        --------
        list
            List of players with their Impact Score, sorted in descending order
        """
        # If specific match requested
        if match_id is not None and str(match_id) in self.smart_metrics:
            match_data = self.smart_metrics[str(match_id)]
            player_impact = match_data.get('player_impact', {})
            
            # Format the rankings
            rankings = []
            for player, impact in player_impact.items():
                rankings.append({
                    'player': player,
                    'player_id': impact.get('player_id'),
                    'batting_impact': impact.get('batting_impact', 0),
                    'bowling_impact': impact.get('bowling_impact', 0),
                    'total_impact': impact.get('total_impact', 0)
                })
            
            # Sort by total impact
            rankings.sort(key=lambda x: x['total_impact'], reverse=True)
            return rankings
        
        # Aggregate across all matches
        aggregated_impact = {}
        
        # Process each match
        for match_id, match_data in self.smart_metrics.items():
            player_impact = match_data.get('player_impact', {})
            
            for player, impact in player_impact.items():
                if player not in aggregated_impact:
                    aggregated_impact[player] = {
                        'player': player,
                        'player_id': impact.get('player_id'),
                        'batting_impact': 0,
                        'bowling_impact': 0,
                        'total_impact': 0,
                        'matches': 0
                    }
                
                # Aggregate impact
                aggregated_impact[player]['batting_impact'] += impact.get('batting_impact', 0)
                aggregated_impact[player]['bowling_impact'] += impact.get('bowling_impact', 0)
                aggregated_impact[player]['total_impact'] += impact.get('total_impact', 0)
                aggregated_impact[player]['matches'] += 1
        
        # Convert to list and sort
        rankings = list(aggregated_impact.values())
        rankings.sort(key=lambda x: x['total_impact'], reverse=True)
        
        return rankings
    
    def get_key_moments(self, match_id):
        """
        Get key moments for a specific match
        
        Parameters:
        -----------
        match_id : float
            Match ID to analyze
        
        Returns:
        --------
        list
            List of key moments in the match
        """
        if str(match_id) not in self.smart_metrics:
            return []
            
        match_data = self.smart_metrics[str(match_id)]
        return match_data.get('key_moments', [])
    
    def get_batsman_smart_stats(self, batsman_name=None, batsman_id=None):
        """
        Get Smart Stats for a specific batsman across all matches
        
        Parameters:
        -----------
        batsman_name : str
            Batsman name
        batsman_id : float
            Batsman ID (preferred over name if both provided)
        
        Returns:
        --------
        dict
            Dictionary with batsman's Smart Stats
        """
        if batsman_id is None and batsman_name is None:
            return None
            
        # Find batsman across all matches
        batsman_stats = {
            'matches': 0,
            'smart_runs_total': 0,
            'conventional_runs_total': 0,
            'balls_faced_total': 0,
            'impact_total': 0,
            'match_performances': []
        }
        
        for match_id, match_data in self.smart_metrics.items():
            # Check in smart runs data
            if 'smart_runs' in match_data:
                for player, stats in match_data['smart_runs'].items():
                    player_id = stats.get('player_id')
                    
                    # Match by ID or name
                    if ((batsman_id is not None and player_id == batsman_id) or
                        (batsman_name is not None and player == batsman_name)):
                        
                        # Add to totals
                        batsman_stats['matches'] += 1
                        batsman_stats['smart_runs_total'] += stats.get('smart_runs', 0)
                        batsman_stats['conventional_runs_total'] += stats.get('conventional_runs', 0)
                        batsman_stats['balls_faced_total'] += stats.get('balls_faced', 0)
                        
                        # Get impact if available
                        impact = 0
                        if 'player_impact' in match_data and player in match_data['player_impact']:
                            impact = match_data['player_impact'][player].get('batting_impact', 0)
                            batsman_stats['impact_total'] += impact
                        
                        # Add match performance
                        batsman_stats['match_performances'].append({
                            'match_id': match_id,
                            'smart_runs': stats.get('smart_runs', 0),
                            'conventional_runs': stats.get('conventional_runs', 0),
                            'balls_faced': stats.get('balls_faced', 0),
                            'smart_sr': stats.get('smart_sr', 0),
                            'conventional_sr': stats.get('conventional_sr', 0),
                            'impact': impact
                        })
        
        # Calculate overall stats
        if batsman_stats['balls_faced_total'] > 0:
            batsman_stats['smart_sr_overall'] = (batsman_stats['smart_runs_total'] / 
                                                 batsman_stats['balls_faced_total']) * 100
            batsman_stats['conventional_sr_overall'] = (batsman_stats['conventional_runs_total'] / 
                                                        batsman_stats['balls_faced_total']) * 100
        else:
            batsman_stats['smart_sr_overall'] = 0
            batsman_stats['conventional_sr_overall'] = 0
            
        batsman_stats['avg_impact_per_match'] = (batsman_stats['impact_total'] / 
                                                batsman_stats['matches']) if batsman_stats['matches'] > 0 else 0
        
        return batsman_stats
    
    def get_bowler_smart_stats(self, bowler_name=None, bowler_id=None):
        """
        Get Smart Stats for a specific bowler across all matches
        
        Parameters:
        -----------
        bowler_name : str
            Bowler name
        bowler_id : float
            Bowler ID (preferred over name if both provided)
        
        Returns:
        --------
        dict
            Dictionary with bowler's Smart Stats
        """
        if bowler_id is None and bowler_name is None:
            return None
            
        # Find bowler across all matches
        bowler_stats = {
            'matches': 0,
            'smart_wickets_total': 0,
            'conventional_wickets_total': 0,
            'total_overs': 0,
            'total_runs': 0,
            'impact_total': 0,
            'match_performances': []
        }
        
        for match_id, match_data in self.smart_metrics.items():
            # Check in smart wickets data
            if 'smart_wickets' in match_data:
                for player, stats in match_data['smart_wickets'].items():
                    player_id = stats.get('player_id')
                    
                    # Match by ID or name
                    if ((bowler_id is not None and player_id == bowler_id) or
                        (bowler_name is not None and player == bowler_name)):
                        
                        # Add to totals
                        bowler_stats['matches'] += 1
                        bowler_stats['smart_wickets_total'] += stats.get('smart_wickets', 0)
                        bowler_stats['conventional_wickets_total'] += stats.get('conventional_wickets', 0)
                        bowler_stats['total_overs'] += stats.get('total_overs', 0)
                        bowler_stats['total_runs'] += stats.get('total_runs', 0)
                        
                        # Get impact if available
                        impact = 0
                        if 'player_impact' in match_data and player in match_data['player_impact']:
                            impact = match_data['player_impact'][player].get('bowling_impact', 0)
                            bowler_stats['impact_total'] += impact
                        
                        # Add match performance
                        bowler_stats['match_performances'].append({
                            'match_id': match_id,
                            'smart_wickets': stats.get('smart_wickets', 0),
                            'conventional_wickets': stats.get('conventional_wickets', 0),
                            'overs': stats.get('total_overs', 0),
                            'runs': stats.get('total_runs', 0),
                            'smart_er': stats.get('smart_er', 0),
                            'conventional_er': stats.get('conventional_er', 0),
                            'impact': impact
                        })
        
        # Calculate overall stats
        if bowler_stats['total_overs'] > 0:
            bowler_stats['conventional_er_overall'] = bowler_stats['total_runs'] / bowler_stats['total_overs']
            
            # Smart economy rate
            if bowler_stats['conventional_wickets_total'] > 0:
                smart_ratio = bowler_stats['smart_wickets_total'] / bowler_stats['conventional_wickets_total']
                smart_runs_conceded = bowler_stats['total_runs'] / smart_ratio
                bowler_stats['smart_er_overall'] = smart_runs_conceded / bowler_stats['total_overs']
            else:
                bowler_stats['smart_er_overall'] = bowler_stats['conventional_er_overall']
        else:
            bowler_stats['conventional_er_overall'] = 0
            bowler_stats['smart_er_overall'] = 0
            
        bowler_stats['avg_impact_per_match'] = (bowler_stats['impact_total'] / 
                                               bowler_stats['matches']) if bowler_stats['matches'] > 0 else 0
        
        return bowler_stats