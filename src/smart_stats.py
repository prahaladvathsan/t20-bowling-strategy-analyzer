import pandas as pd
import numpy as np
import math
import json
from pathlib import Path
from data_processor import DataProcessor

class PressureIndexEngine:
    """
    Implements the ESPNcricinfo Pressure Index calculation
    """
    
    # Wicket weighting scale from ESPNcricinfo guide
    WICKET_WEIGHTS = {
        1: 0.5, 2: 0.7, 3: 0.9, 4: 1.1, 5: 1.3,
        6: 1.5, 7: 1.7, 8: 1.9, 9: 2.1, 10: 2.3
    }
    
    # Phase multipliers for different game phases
    PHASE_MULTIPLIERS = {
        1: 1.2,  # Powerplay
        2: 1.0,  # Middle overs
        3: 1.5   # Death overs
    }
    
    def __init__(self, data=None, player_quality=None):
        """Initialize with ball-by-ball data and supporting datasets"""
        self.data = data
        self.player_quality = player_quality or {}
        
    def calculate_pressure_index(self, ball_data):
        """
        Calculate pressure index for a single ball based on your dataset structure
        
        Parameters:
        -----------
        ball_data : pandas.Series
            Series with ball information
        
        Returns:
        --------
        float
            Pressure index value
        """
        # Your data already has many components needed for the calculation
        
        # Required run rate (using inns_rrr if available)
        if 'inns_rrr' in ball_data and not pd.isna(ball_data['inns_rrr']):
            rrr = ball_data['inns_rrr']
        elif 'target' in ball_data and 'inns_balls_rem' in ball_data and ball_data['inns_balls_rem'] > 0:
            runs_needed = ball_data['target'] - ball_data['inns_runs']
            rrr = (runs_needed / ball_data['inns_balls_rem']) * 6
        else:
            rrr = 1.0  # Default value
            
        # Initial required run rate
        if 'target' in ball_data and 'max_balls' in ball_data and ball_data['max_balls'] > 0:
            irr = (ball_data['target'] / (ball_data['max_balls'] / 6))
        else:
            irr = 8.0  # Default T20 run rate
            
        # Wicket weight based on current wickets
        wickets_lost = ball_data['inns_wkts'] if 'inns_wkts' in ball_data else 0
        wicket_weight = self.WICKET_WEIGHTS.get(int(wickets_lost) if not pd.isna(wickets_lost) else 0, 0)
        
        # Balls remaining factor
        if 'inns_balls_rem' in ball_data and 'max_balls' in ball_data:
            balls_remaining = ball_data['inns_balls_rem']
            total_balls = ball_data['max_balls']
            br_factor = math.sqrt(balls_remaining / total_balls) if total_balls > 0 else 0.5
        else:
            br_factor = 0.5  # Default middle-of-innings value
            
        # Quality modifier using player IDs
        bat_id = ball_data['p_bat'] if 'p_bat' in ball_data else None
        bowl_id = ball_data['p_bowl'] if 'p_bowl' in ball_data else None
        
        # Get player qualities (or default values)
        bat_quality = self.player_quality.get(bat_id, {}).get('batting_quality', 100) if bat_id else 100
        bowl_quality = self.player_quality.get(bowl_id, {}).get('bowling_quality', 8) if bowl_id else 8
        
        # Normalize qualities (higher batting quality and lower bowling economy are better)
        qm = (bat_quality / 150) * (8 / bowl_quality) if bowl_quality > 0 else 1
        
        # Calculate final pressure index
        try:
            pressure_index = (rrr / irr if irr > 0 else 1) * (1 + wicket_weight/10) * br_factor * qm
            
            # Cap at reasonable values and handle NaN
            if np.isnan(pressure_index) or pressure_index > 10:
                pressure_index = 1.0
                
            return pressure_index
        except Exception:
            return 1.0  # Default in case of calculation errors
    
    def process_match(self, match_id):
        """
        Process an entire match to calculate pressure indices for each ball
        
        Parameters:
        -----------
        match_id : float
            Match identifier
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with original data and added pressure indices
        """
        if self.data is None:
            raise ValueError("No data loaded")
            
        # Filter data for the specific match
        match_data = self.data[self.data['p_match'] == match_id].copy()
        if len(match_data) == 0:
            return pd.DataFrame()  # No data for this match
            
        match_data = match_data.sort_values(['inns', 'over', 'ball'])
        
        # Add pressure index column
        match_data['pressure_index'] = 0.0
        
        # Process each ball
        for idx, ball in match_data.iterrows():
            # Calculate pressure index
            match_data.at[idx, 'pressure_index'] = self.calculate_pressure_index(ball)
                
        return match_data


class SmartStatsCalculator:
    """
    Calculates Smart Stats metrics including Smart Runs and Smart Wickets
    """
    
    def __init__(self, data=None, pressure_engine=None, player_quality=None):
        """Initialize with ball-by-ball data and supporting components"""
        self.data = data
        self.pressure_engine = pressure_engine
        self.player_quality = player_quality or {}
        self.smart_metrics = {}
        
    def calculate_smart_runs(self, match_id):
        """
        Calculate Smart Runs for all batsmen in a match
        
        Parameters:
        -----------
        match_id : float
            Match identifier
        
        Returns:
        --------
        dict
            Dictionary with batsmen and their Smart Runs values
        """
        if self.data is None or self.pressure_engine is None:
            raise ValueError("Data or pressure engine not initialized")
            
        # Get match data with pressure indices
        match_data = self.pressure_engine.process_match(match_id)
        if len(match_data) == 0:
            return {}  # No data for this match
            
        # Initialize smart runs dictionary
        smart_runs = {}
        
        # Process each ball
        for _, ball in match_data.iterrows():
            # Skip wide balls or no balls for batsmen stats
            if (('wide' in ball and ball['wide'] > 0) or 
                ('noball' in ball and ball['noball'] > 0 and ball['score'] == 0)):
                continue
                
            # Get batsman info
            batsman = ball['bat']
            batsman_id = ball['p_bat'] if 'p_bat' in ball else None
            
            # Skip if no valid batsman
            if pd.isna(batsman) or batsman == '':
                continue
                
            # Get runs and pressure
            runs = ball['score'] if 'score' in ball and not pd.isna(ball['score']) else 0
            pressure = ball['pressure_index']
            
            # Basic smart runs calculation: runs * pressure
            base_value = runs * pressure
            
            # Quality adjustment based on bowling opposition
            bowler = ball['bowl'] if 'bowl' in ball else ''
            bowl_id = ball['p_bowl'] if 'p_bowl' in ball else None
            
            bowl_quality = self.player_quality.get(bowl_id, {}).get('bowling_quality', 8) if bowl_id else 8
            difficulty = 1 + (bowl_quality / 100)
            
            # Calculate final smart runs value
            smart_value = base_value * difficulty
            
            # Add to batsman's total
            if batsman not in smart_runs:
                smart_runs[batsman] = {
                    'player_id': batsman_id,
                    'smart_runs': 0,
                    'conventional_runs': 0,
                    'balls_faced': 0
                }
                
            smart_runs[batsman]['smart_runs'] += smart_value
            smart_runs[batsman]['conventional_runs'] += runs
            smart_runs[batsman]['balls_faced'] += 1
            
        # Calculate strike rates
        for batsman, stats in smart_runs.items():
            if stats['balls_faced'] > 0:
                stats['conventional_sr'] = (stats['conventional_runs'] / stats['balls_faced']) * 100
                stats['smart_sr'] = (stats['smart_runs'] / stats['balls_faced']) * 100
            else:
                stats['conventional_sr'] = 0
                stats['smart_sr'] = 0
            
        return smart_runs
    
    def calculate_smart_wickets(self, match_id):
        """
        Calculate Smart Wickets for all bowlers in a match
        
        Parameters:
        -----------
        match_id : float
            Match identifier
        
        Returns:
        --------
        dict
            Dictionary with bowlers and their Smart Wickets values
        """
        if self.data is None:
            raise ValueError("Data not initialized")
            
        # Get match data with pressure indices
        match_data = self.pressure_engine.process_match(match_id)
        if len(match_data) == 0:
            return {}  # No data for this match
            
        # Initialize smart wickets dictionary
        smart_wickets = {}
        
        # Process each dismissal
        dismissals = match_data[match_data['out'] == 'TRUE']
        
        for _, dismissal in dismissals.iterrows():
            # Get bowler info
            bowler = dismissal['bowl'] if 'bowl' in dismissal else ''
            bowler_id = dismissal['p_bowl'] if 'p_bowl' in dismissal else None
            
            # Skip if no valid bowler (run outs, etc.)
            if pd.isna(bowler) or bowler == '':
                continue
                
            # Get batsman info
            batsman = dismissal['bat'] if 'bat' in dismissal else ''
            batsman_id = dismissal['p_bat'] if 'p_bat' in dismissal else None
            
            # Calculate base components from formula:
            # Wicket Value = (0.4*batsman_quality + 0.3*phase_multiplier + 0.3*win_impact) * 10
            
            # Batsman quality component
            bat_quality = self.player_quality.get(batsman_id, {}).get('batting_quality', 100) if batsman_id else 100
            batsman_quality = bat_quality / 150  # Normalize to 0-1 range
            
            # Phase multiplier
            phase = dismissal['phase'] if 'phase' in dismissal and not pd.isna(dismissal['phase']) else 2
            phase_multiplier = self.pressure_engine.PHASE_MULTIPLIERS.get(phase, 1.0)
            
            # Win impact - use win probability change if available
            if 'wprob' in dismissal and not pd.isna(dismissal['wprob']):
                # Find the next ball to get win prob change
                current_idx = dismissal.name
                next_ball = match_data.loc[current_idx+1] if current_idx+1 in match_data.index else None
                
                if next_ball is not None and 'wprob' in next_ball and not pd.isna(next_ball['wprob']):
                    win_impact = abs(next_ball['wprob'] - dismissal['wprob'])
                else:
                    win_impact = 0.5  # Default if we can't compute change
            else:
                win_impact = 0.5  # Default value
            
            # Calculate wicket value
            wicket_value = (0.4 * batsman_quality + 0.3 * phase_multiplier + 0.3 * win_impact) * 10
            
            # Add to bowler's total
            if bowler not in smart_wickets:
                # Calculate total overs
                bowler_balls = match_data[match_data['bowl'] == bowler].shape[0]
                bowler_overs = bowler_balls / 6
                
                smart_wickets[bowler] = {
                    'player_id': bowler_id,
                    'smart_wickets': 0,
                    'conventional_wickets': 0,
                    'total_overs': bowler_overs,
                    'total_runs': match_data[match_data['bowl'] == bowler]['score'].sum()
                }
                
            smart_wickets[bowler]['smart_wickets'] += wicket_value
            smart_wickets[bowler]['conventional_wickets'] += 1
            
        # Calculate economy rates
        for bowler, stats in smart_wickets.items():
            if stats['total_overs'] > 0:
                stats['conventional_er'] = stats['total_runs'] / stats['total_overs']
                
                # Smart economy uses the same ratio of smart wickets to conventional wickets
                if stats['conventional_wickets'] > 0:
                    smart_ratio = stats['smart_wickets'] / stats['conventional_wickets']
                    smart_runs_conceded = stats['total_runs'] / smart_ratio
                    stats['smart_er'] = smart_runs_conceded / stats['total_overs']
                else:
                    stats['smart_er'] = stats['conventional_er']
            else:
                stats['conventional_er'] = 0
                stats['smart_er'] = 0
            
        return smart_wickets
    
    def calculate_player_impact(self, match_id):
        """
        Calculate overall Player Impact for all players in a match
        
        Parameters:
        -----------
        match_id : float
            Match identifier
        
        Returns:
        --------
        dict
            Dictionary with players and their impact values
        """
        # Get smart runs and wickets first
        smart_runs = self.calculate_smart_runs(match_id)
        smart_wickets = self.calculate_smart_wickets(match_id)
        
        # Initialize player impact dictionary
        player_impact = {}
        
        # Combine batting and bowling impact
        all_players = set(list(smart_runs.keys()) + list(smart_wickets.keys()))
        
        for player in all_players:
            # Get batting impact
            batting_stats = smart_runs.get(player, {})
            batting_impact = batting_stats.get('smart_runs', 0) / 30  # Normalize impact
            
            # Get bowling impact
            bowling_stats = smart_wickets.get(player, {})
            bowling_impact = bowling_stats.get('smart_wickets', 0) / 5  # Normalize impact
            
            # Get player ID (prefer batsman ID, fall back to bowler ID)
            player_id = (batting_stats.get('player_id') or 
                         bowling_stats.get('player_id'))
            
            player_impact[player] = {
                'player_id': player_id,
                'batting_impact': batting_impact,
                'bowling_impact': bowling_impact,
                'total_impact': batting_impact + bowling_impact
            }
            
        return player_impact
    
    def identify_key_moments(self, match_id):
        """
        Identify key moments in a match based on impact on win probability
        
        Parameters:
        -----------
        match_id : float
            Match identifier
        
        Returns:
        --------
        list
            List of key moments with player attribution
        """
        if self.data is None:
            raise ValueError("Data not initialized")
            
        # Get match data
        match_data = self.data[self.data['p_match'] == match_id].copy()
        if len(match_data) == 0 or 'wprob' not in match_data.columns:
            return []  # No data or no win probability data
        
        # Calculate win probability changes
        match_data = match_data.sort_values(['inns', 'over', 'ball'])
        match_data['wprob_shift'] = match_data['wprob'].shift(-1) - match_data['wprob']
        
        # Find significant moments (>= 10% win probability change)
        significant = match_data[abs(match_data['wprob_shift']) >= 0.1].copy()
        
        key_moments = []
        
        for _, ball in significant.iterrows():
            # Determine if batting or bowling impact
            is_batting_impact = True
            
            # If it's a wicket, it's a bowling impact
            if 'out' in ball and ball['out'] == 'TRUE':
                is_batting_impact = False
            
            # For boundaries, it's batting impact
            if 'score' in ball and ball['score'] >= 4:
                is_batting_impact = True
                
            # Create moment description
            over = ball['over'] if 'over' in ball else 0
            ball_num = ball['ball'] if 'ball' in ball else 0
            runs = ball['score'] if 'score' in ball else 0
            
            # Format description based on event type
            if 'out' in ball and ball['out'] == 'TRUE':
                description = f"Wicket at {over:.1f}: {ball['bowl']} dismissed {ball['bat']}"
                primary_player = ball['bowl']
                player_id = ball['p_bowl'] if 'p_bowl' in ball else None
            elif 'score' in ball and ball['score'] >= 4:
                description = f"Boundary at {over:.1f}: {ball['bat']} hit {int(runs)} runs"
                primary_player = ball['bat']
                player_id = ball['p_bat'] if 'p_bat' in ball else None
            else:
                description = f"Impact play at {over:.1f}: Changed win probability by {abs(ball['wprob_shift'])*100:.1f}%"
                primary_player = ball['bat'] if is_batting_impact else ball['bowl']
                player_id = ball['p_bat'] if is_batting_impact else ball['p_bowl']
            
            # Add to key moments
            key_moments.append({
                'match_id': match_id,
                'ball_id': ball['ball_id'] if 'ball_id' in ball else None,
                'over': over,
                'player': primary_player,
                'player_id': player_id,
                'description': description,
                'delta_win_prob': ball['wprob_shift'],
                'batting_impact': ball['wprob_shift'] if is_batting_impact else 0,
                'bowling_impact': ball['wprob_shift'] if not is_batting_impact else 0,
                'total_impact': ball['wprob_shift']
            })
            
        # Sort by impact magnitude
        key_moments.sort(key=lambda x: abs(x['delta_win_prob']), reverse=True)
        
        return key_moments[:10]  # Return top 10 moments
    
    def calculate_smart_metrics_for_all_matches(self):
        """Calculate Smart Metrics for all matches in the dataset"""
        if self.data is None:
            raise ValueError("Data not initialized")
            
        all_matches = pd.unique(self.data['p_match']).tolist()
        
        for match_id in all_matches:
            if pd.isna(match_id):
                continue
                
            smart_runs = self.calculate_smart_runs(match_id)
            smart_wickets = self.calculate_smart_wickets(match_id)
            player_impact = self.calculate_player_impact(match_id)
            key_moments = self.identify_key_moments(match_id)
            
            # Store in smart_metrics dictionary
            self.smart_metrics[match_id] = {
                'smart_runs': smart_runs,
                'smart_wickets': smart_wickets,
                'player_impact': player_impact,
                'key_moments': key_moments
            }
            
        return self.smart_metrics


class PlayerQualityCalculator:
    """
    Calculates player quality indices for use in Smart Stats calculations
    """
    
    def __init__(self, data=None):
        """Initialize with historical player data"""
        self.data = data
        self.player_quality = {}
        
    def calculate_batting_quality(self, player_id, player_name=None, recent_matches=10):
        """
        Calculate batting quality index
        
        Formula: 0.4*CareerSR + 0.3*RecentForm + 0.3*VenueAvg
        """
        if self.data is None:
            raise ValueError("Data not initialized")
            
        # Find by ID first, then by name if needed
        if player_id is not None:
            player_data = self.data[self.data['p_bat'] == player_id]
        elif player_name is not None:
            player_data = self.data[self.data['bat'] == player_name]
        else:
            return 100  # Default quality
            
        if len(player_data) == 0:
            return 100  # Default quality
            
        # Career strike rate
        career_runs = player_data['score'].sum()
        career_balls = len(player_data)
        career_sr = (career_runs / career_balls) * 100 if career_balls > 0 else 100
        
        # Recent form (last N matches)
        recent_matches_data = player_data.sort_values('p_match', ascending=False)
        recent_matches_list = pd.unique(recent_matches_data['p_match'])[:recent_matches]
        recent_data = player_data[player_data['p_match'].isin(recent_matches_list)]
        
        recent_runs = recent_data['score'].sum()
        recent_balls = len(recent_data)
        recent_form = (recent_runs / recent_balls) * 100 if recent_balls > 0 else career_sr
        
        # Venue average (simplified - group by ground)
        venue_sr = career_sr  # Default to career SR
        
        if 'ground' in player_data.columns:
            venue_data = player_data.groupby('ground').agg({
                'score': 'sum',
                'bat': 'count'  # Using count of bat as proxy for balls faced
            })
            
            # Calculate weighted average across venues
            total_balls = venue_data['bat'].sum()
            if total_balls > 0:
                venue_sr = sum((venue_data['score'] / venue_data['bat']) * venue_data['bat']) / total_balls * 100
        
        # Calculate final quality index
        quality = 0.4 * career_sr + 0.3 * recent_form + 0.3 * venue_sr
        
        return quality
    
    def calculate_bowling_quality(self, player_id, player_name=None, recent_matches=10):
        """
        Calculate bowling quality index
        
        Formula: 0.5*CareerER + 0.3*RecentForm + 0.2*VenueER
        """
        if self.data is None:
            raise ValueError("Data not initialized")
            
        # Find by ID first, then by name if needed
        if player_id is not None:
            player_data = self.data[self.data['p_bowl'] == player_id]
        elif player_name is not None:
            player_data = self.data[self.data['bowl'] == player_name]
        else:
            return 8.0  # Default economy rate
            
        if len(player_data) == 0:
            return 8.0  # Default economy rate
            
        # Career economy rate
        career_runs = player_data['score'].sum()
        career_balls = len(player_data)
        career_overs = career_balls / 6
        career_er = (career_runs / career_overs) if career_overs > 0 else 8.0
        
        # Recent form (last N matches)
        recent_matches_data = player_data.sort_values('p_match', ascending=False)
        recent_matches_list = pd.unique(recent_matches_data['p_match'])[:recent_matches]
        recent_data = player_data[player_data['p_match'].isin(recent_matches_list)]
        
        recent_runs = recent_data['score'].sum()
        recent_balls = len(recent_data)
        recent_overs = recent_balls / 6
        recent_form = (recent_runs / recent_overs) if recent_overs > 0 else career_er
        
        # Venue economy (simplified - group by ground)
        venue_er = career_er  # Default to career ER
        
        if 'ground' in player_data.columns:
            venue_data = player_data.groupby('ground').agg({
                'score': 'sum',
                'bowl': 'count'  # Using count of bowl as proxy for balls bowled
            })
            
            # Calculate weighted average across venues
            total_balls = venue_data['bowl'].sum()
            venue_overs = total_balls / 6
            if venue_overs > 0:
                venue_er = venue_data['score'].sum() / venue_overs
        
        # Calculate final quality index (lower is better for bowlers)
        quality = 0.5 * career_er + 0.3 * recent_form + 0.2 * venue_er
        
        return quality
    
    def calculate_player_qualities(self):
        """Calculate quality indices for all players in the dataset"""
        if self.data is None:
            raise ValueError("Data not initialized")
            
        # Get all unique players
        batsmen_ids = pd.unique(self.data['p_bat']).tolist()
        bowler_ids = pd.unique(self.data['p_bowl']).tolist()
        
        # Calculate for batsmen
        for player_id in batsmen_ids:
            if pd.isna(player_id):
                continue
                
            player_rows = self.data[self.data['p_bat'] == player_id]
            player_name = player_rows['bat'].iloc[0] if len(player_rows) > 0 else None
            
            batting_quality = self.calculate_batting_quality(player_id)
            
            if player_id not in self.player_quality:
                self.player_quality[player_id] = {}
                
            self.player_quality[player_id]['player_name'] = player_name
            self.player_quality[player_id]['batting_quality'] = batting_quality
            
        # Calculate for bowlers
        for player_id in bowler_ids:
            if pd.isna(player_id):
                continue
                
            player_rows = self.data[self.data['p_bowl'] == player_id]
            player_name = player_rows['bowl'].iloc[0] if len(player_rows) > 0 else None
            
            bowling_quality = self.calculate_bowling_quality(player_id)
            
            if player_id not in self.player_quality:
                self.player_quality[player_id] = {}
                
            self.player_quality[player_id]['player_name'] = player_name
            self.player_quality[player_id]['bowling_quality'] = bowling_quality
            
        return self.player_quality