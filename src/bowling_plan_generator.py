# src/bowling_plan_generator.py

import pandas as pd
import numpy as np
from src.data_processor import DataProcessor

class BowlingPlanGenerator:
    """
    Generates comprehensive bowling plans against specific batters
    based on ball-by-ball data analysis
    """
    
    def __init__(self, data):
        """Initialize with ball-by-ball dataset"""
        self.data = data
        self.batter_profiles = self._create_batter_profiles()
        self.bowler_profiles = self._create_bowler_profiles()
        self.phase_insights = self._analyze_game_phases()
        
    def _create_batter_profiles(self):
        """
        Create profiles for batters based on their performance against
        different bowling types, lines, and lengths
        """
        profiles = {}
        
        # Group by batter
        batters = self.data['bat'].unique()
        
        for batter in batters:
            batter_data = self.data[self.data['bat'] == batter]
            
            # Get batter hand
            try:
                bat_hand = batter_data['bat_hand'].iloc[0]
            except:
                bat_hand = 'Unknown'
            
            # Calculate overall stats
            total_runs = batter_data['score'].sum()
            total_balls = len(batter_data)
            dismissals = batter_data['out'].sum()
            
            # Strike rate and average
            strike_rate = (total_runs / total_balls * 100) if total_balls > 0 else 0
            average = (total_runs / dismissals) if dismissals > 0 else float('inf')
            
            # Analysis by bowling type
            bowl_kinds = batter_data.groupby('bowl_kind').agg({
                'score': 'sum',
                'out': 'sum'
            })
            
            bowl_kind_stats = {}
            for kind, stats in bowl_kinds.iterrows():
                runs = stats['score']
                balls = len(batter_data[batter_data['bowl_kind'] == kind])
                outs = stats['out']
                
                sr = (runs / balls * 100) if balls > 0 else 0
                avg = (runs / outs) if outs > 0 else float('inf')
                
                bowl_kind_stats[kind] = {
                    'runs': runs,
                    'balls': balls,
                    'dismissals': outs,
                    'strike_rate': sr,
                    'average': avg
                }
            
            # Analysis by line and length
            line_length_stats = {}
            
            if 'line' in batter_data.columns and 'length' in batter_data.columns:
                for line in batter_data['line'].unique():
                    for length in batter_data['length'].unique():
                        if line == 'Unknown' or length == 'Unknown':
                            continue
                            
                        line_length_data = batter_data[
                            (batter_data['line'] == line) & 
                            (batter_data['length'] == length)
                        ]
                        
                        if len(line_length_data) > 0:
                            runs = line_length_data['score'].sum()
                            balls = len(line_length_data)
                            outs = line_length_data['out'].sum()
                            
                            sr = (runs / balls * 100) if balls > 0 else 0
                            avg = (runs / outs) if outs > 0 else float('inf')
                            
                            line_length_stats[(line, length)] = {
                                'runs': runs,
                                'balls': balls,
                                'dismissals': outs,
                                'strike_rate': sr,
                                'average': avg
                            }
            
            # Analysis by game phase
            phase_stats = {}
            
            if 'phase' in batter_data.columns:
                for phase in batter_data['phase'].unique():
                    phase_data = batter_data[batter_data['phase'] == phase]
                    
                    if len(phase_data) > 0:
                        runs = phase_data['score'].sum()
                        balls = len(phase_data)
                        outs = phase_data['out'].sum()
                        
                        sr = (runs / balls * 100) if balls > 0 else 0
                        avg = (runs / outs) if outs > 0 else float('inf')
                        
                        phase_stats[phase] = {
                            'runs': runs,
                            'balls': balls,
                            'dismissals': outs,
                            'strike_rate': sr,
                            'average': avg
                        }
            
            # Store profile
            profiles[batter] = {
                'bat_hand': bat_hand,
                'total_runs': total_runs,
                'total_balls': total_balls,
                'dismissals': dismissals,
                'strike_rate': strike_rate,
                'average': average,
                'vs_bowler_types': bowl_kind_stats,
                'vs_line_length': line_length_stats,
                'by_phase': phase_stats
            }
        
        return profiles
    
    def _create_bowler_profiles(self):
        """
        Create profiles for bowler types based on their performance with
        different lines and lengths
        """
        profiles = {}
        
        # Group by bowler type
        bowler_types = self.data['bowl_kind'].unique()
        
        for bowl_kind in bowler_types:
            if pd.isna(bowl_kind) or bowl_kind == 'Unknown':
                continue
                
            bowler_data = self.data[self.data['bowl_kind'] == bowl_kind]
            
            # Calculate overall stats
            total_runs = bowler_data['score'].sum()
            total_balls = len(bowler_data)
            wickets = bowler_data['out'].sum()
            
            # Economy and average
            economy = (total_runs / total_balls * 6) if total_balls > 0 else 0
            average = (total_runs / wickets) if wickets > 0 else float('inf')
            strike_rate = (total_balls / wickets) if wickets > 0 else float('inf')
            
            # Store profile
            profiles[bowl_kind] = {
                'total_runs': total_runs,
                'total_balls': total_balls,
                'wickets': wickets,
                'economy': economy,
                'average': average,
                'strike_rate': strike_rate
            }
        
        return profiles
    
    def _analyze_game_phases(self):
        """
        Analyze how game phases affect bowling and batting performance
        """
        phase_insights = {}
        
        if 'phase' in self.data.columns:
            for phase in self.data['phase'].unique():
                phase_data = self.data[self.data['phase'] == phase]
                
                if len(phase_data) > 0:
                    runs = phase_data['score'].sum()
                    balls = len(phase_data)
                    wickets = phase_data['out'].sum()
                    
                    run_rate = (runs / balls * 6) if balls > 0 else 0
                    wicket_rate = (wickets / balls * 6) if balls > 0 else 0
                    
                    phase_insights[phase] = {
                        'run_rate': run_rate,
                        'wicket_rate': wicket_rate,
                        'runs_per_wicket': (runs / wickets) if wickets > 0 else float('inf')
                    }
        
        return phase_insights
    
    def find_batter_weaknesses(self, batter):
        """
        Identify weaknesses for a specific batter
        
        Parameters:
        -----------
        batter : str
            Batter name
            
        Returns:
        --------
        Dictionary with identified weaknesses
        """
        if batter not in self.batter_profiles:
            return {"error": "Batter not found in dataset"}
        
        profile = self.batter_profiles[batter]
        weaknesses = []
        
        # Check weakness against bowler types
        if len(profile['vs_bowler_types']) > 0:
            # Find bowler type with lowest average
            best_bowler_types = []
            
            for kind, stats in profile['vs_bowler_types'].items():
                if stats['dismissals'] > 0 and stats['balls'] >= 5:
                    best_bowler_types.append((kind, stats['average']))
            
            # Sort by average (lower is better)
            best_bowler_types.sort(key=lambda x: x[1])
            
            if best_bowler_types and best_bowler_types[0][1] < profile['average'] * 0.8:
                kind, avg = best_bowler_types[0]
                weaknesses.append({
                    'type': 'bowler_type',
                    'bowler_type': kind,
                    'average': avg,
                    'overall_average': profile['average'],
                    'confidence': 'high' if profile['vs_bowler_types'][kind]['balls'] > 30 else 'medium'
                })
        
        # Check weakness against line and length
        if len(profile['vs_line_length']) > 0:
            # Find line-length with lowest average
            best_line_lengths = []
            
            for combo, stats in profile['vs_line_length'].items():
                if stats['dismissals'] > 0 and stats['balls'] >= 5:
                    best_line_lengths.append((combo, stats['average']))
            
            # Sort by average (lower is better)
            best_line_lengths.sort(key=lambda x: x[1])
            
            if best_line_lengths and best_line_lengths[0][1] < profile['average'] * 0.7:
                combo, avg = best_line_lengths[0]
                line_num, length_num = combo
                line_display = DataProcessor.LINE_DISPLAY.get(line_num, 'Unknown')
                length_display = DataProcessor.LENGTH_DISPLAY.get(length_num, 'Unknown')

                weaknesses.append({
                    'type': 'line_length',
                    'line': line_display,
                    'length': length_display,
                    'average': avg,
                    'overall_average': profile['average'],
                    'confidence': 'high' if profile['vs_line_length'][combo]['balls'] > 20 else 'medium'
                })
        
        # Check weakness by phase
        if len(profile['by_phase']) > 0:
            # Find phase with lowest strike rate
            worst_phases = []
            
            for phase, stats in profile['by_phase'].items():
                if stats['balls'] >= 10:
                    worst_phases.append((phase, stats['strike_rate']))
            
            # Sort by strike rate (lower is worse)
            worst_phases.sort(key=lambda x: x[1])
            
            if worst_phases and worst_phases[0][1] < profile['strike_rate'] * 0.8:
                phase, sr = worst_phases[0]
                weaknesses.append({
                    'type': 'phase',
                    'phase': phase,
                    'strike_rate': sr,
                    'overall_strike_rate': profile['strike_rate'],
                    'confidence': 'high' if profile['by_phase'][phase]['balls'] > 30 else 'medium'
                })
        
        return {
            'batter': batter,
            'bat_hand': profile['bat_hand'],
            'overall_average': profile['average'],
            'overall_strike_rate': profile['strike_rate'],
            'weaknesses': weaknesses
        }
    
    def identify_optimal_matchups(self, batter, available_bowlers=None):
        """
        Identify optimal bowler matchups against a specific batter
        
        Parameters:
        -----------
        batter : str
            Batter name
            
        available_bowlers : list of dict, optional
            List of available bowlers with their types
            
        Returns:
        --------
        Dictionary with optimal matchups
        """
        if batter not in self.batter_profiles:
            return {"error": "Batter not found in dataset"}
        
        # Extract batter's performance against different bowler types
        profile = self.batter_profiles[batter]
        bowler_type_performance = profile['vs_bowler_types']
        
        # If no specific bowlers provided, evaluate all bowler types
        if available_bowlers is None:
            available_types = list(self.bowler_profiles.keys())
            available_bowlers = [{'type': t} for t in available_types]
        
        # Evaluate each available bowler
        matchups = []
        
        for bowler in available_bowlers:
            bowl_type = bowler.get('type', bowler.get('bowl_kind', 'Unknown'))
            
            if bowl_type == 'Unknown' or bowl_type not in self.bowler_profiles:
                continue
            
            # Check if we have data on batter vs this bowler type
            if bowl_type in bowler_type_performance:
                batter_vs_type = bowler_type_performance[bowl_type]
                
                # Only include if we have sufficient data
                if batter_vs_type['balls'] < 5:
                    continue
                
                # Calculate matchup score (lower is better for bowling side)
                # Based on weighted combination of strike rate and average
                if batter_vs_type['dismissals'] > 0:
                    matchup_score = (
                        0.7 * (batter_vs_type['average'] / profile['average'] if profile['average'] > 0 else 1) +
                        0.3 * (batter_vs_type['strike_rate'] / profile['strike_rate'] if profile['strike_rate'] > 0 else 1)
                    )
                else:
                    # If no dismissals, use only strike rate
                    matchup_score = batter_vs_type['strike_rate'] / profile['strike_rate'] if profile['strike_rate'] > 0 else 1
                
                # Calculate confidence level based on sample size
                confidence = 'low'
                if batter_vs_type['balls'] > 50:
                    confidence = 'high'
                elif batter_vs_type['balls'] > 20:
                    confidence = 'medium'
                
                bowler_info = {
                    'bowler': bowler.get('name', bowl_type),
                    'type': bowl_type,
                    'matchup_score': matchup_score,
                    'batter_average': batter_vs_type['average'],
                    'batter_strike_rate': batter_vs_type['strike_rate'],
                    'sample_size': batter_vs_type['balls'],
                    'confidence': confidence,
                    'recommendations': []  # Will be filled below
                }
                
                # Add optimal line and length recommendations
                line_length_recommendations = self._get_line_length_recommendations(batter, bowl_type)
                if line_length_recommendations:
                    bowler_info['recommendations'] = line_length_recommendations
                
                matchups.append(bowler_info)
        
        # Sort matchups by matchup score (lower is better)
        matchups.sort(key=lambda x: x['matchup_score'])
        
        return {
            'batter': batter,
            'matchups': matchups
        }
    
    def _get_line_length_recommendations(self, batter, bowler_type):
        """
        Get line and length recommendations for a specific batter and bowler type
        """
        recommendations = []
        
        # Get batter profile
        if batter not in self.batter_profiles:
            return recommendations
            
        profile = self.batter_profiles[batter]
        line_length_stats = profile.get('vs_line_length', {})
        
        if not line_length_stats:
            return recommendations
        
        # Find all combinations that we have data for
        valid_combinations = []
        
        for combo, stats in line_length_stats.items():
            # Skip if insufficient data
            if stats['balls'] < 5:
                continue
            
            # Calculate effectiveness score
            # Based on strike rate and average
            effectiveness = 0
            
            if stats['dismissals'] > 0:
                # Lower average is better
                relative_avg = stats['average'] / profile.get('average', 30) if profile.get('average', 30) > 0 else 1
                effectiveness += (1 - relative_avg) * 5
            
            # Lower strike rate is better
            relative_sr = stats['strike_rate'] / profile.get('strike_rate', 120) if profile.get('strike_rate', 120) > 0 else 1
            effectiveness += (1 - relative_sr) * 3
            
            # Add recommendation
            valid_combinations.append({
                'line': combo[0],
                'length': combo[1],
                'effectiveness': effectiveness,
                'batter_average': stats['average'] if stats['dismissals'] > 0 else float('inf'),
                'batter_strike_rate': stats['strike_rate'],
                'sample_size': stats['balls'],
                'confidence': 'high' if stats['balls'] > 20 else 'medium'
            })
        
        # Sort by effectiveness (higher is better)
        valid_combinations.sort(key=lambda x: x['effectiveness'], reverse=True)
        
        return valid_combinations[:3]  # Return top 3 recommendations
    
    def generate_phase_specific_plan(self, batter, bowler_types=None):
        """
        Generate a phase-specific bowling plan for a particular batter
        
        Parameters:
        -----------
        batter : str
            Batter name
            
        bowler_types : list
            List of bowler types to consider
            
        Returns:
        --------
        Dictionary with phase-specific plans
        """
        if batter not in self.batter_profiles:
            return {"error": "Batter not found in dataset"}
            
        # If no bowler types specified, use all available types
        if bowler_types is None:
            bowler_types = list(self.bowler_profiles.keys())
        
        plans = {}
        
        # Define phases
        phases = {
            1: 'Powerplay (1-6)',
            2: 'Middle Overs (7-15)',
            3: 'Death Overs (16-20)'
        }
        
        # For each phase
        for phase_id in phases:
            # Get optimal bowler type for this phase
            optimal_bowler = None
            best_score = float('inf')
            
            # Find best bowler type
            batter_profile = self.batter_profiles[batter]
            for bowl_type in bowler_types:
                if bowl_type in batter_profile.get('vs_bowler_types', {}):
                    bowl_stats = batter_profile['vs_bowler_types'][bowl_type]
                    
                    # Calculate a score (lower is better)
                    if bowl_stats['balls'] >= 5:
                        if bowl_stats['dismissals'] > 0:
                            score = bowl_stats['average'] * 0.6 + (bowl_stats['strike_rate'] / 100) * 0.4
                        else:
                            score = bowl_stats['strike_rate'] / 50  # Normalize to similar scale
                        
                        if score < best_score:
                            best_score = score
                            optimal_bowler = bowl_type
            
            # Get optimal line and length combinations
            line_length_recommendations = []
            
            if optimal_bowler:
                # Get specific recommendations
                line_length_recommendations = self._get_line_length_recommendations(batter, optimal_bowler)
            
            # Create field setting based on phase
            field_setting = self._generate_field_setting(phase_id, optimal_bowler, line_length_recommendations)
            
            # Add to plans
            plans[phase_id] = {
                'phase': phases[phase_id],
                'optimal_bowler_type': optimal_bowler,
                'batter_phase_strike_rate': batter_profile.get('by_phase', {}).get(phase_id, {}).get('strike_rate', 'Unknown'),
                'line_length_recommendations': line_length_recommendations,
                'field_setting': field_setting
            }
        
        return {
            'batter': batter,
            'bat_hand': self.batter_profiles[batter].get('bat_hand', 'Unknown'),
            'phase_plans': plans
        }
    
    def _generate_field_setting(self, phase, bowler_type, recommendations):
        """Generate field setting based on phase and bowling plan"""
        field_setting = {
            'catching_positions': [],
            'boundary_riders': [],
            'description': ''
        }
        
        # If we have recommendations, get the top line/length combination
        if recommendations and len(recommendations) > 0:
            top_rec = recommendations[0]
            line = top_rec.get('line')
            length = top_rec.get('length')
            
            # Adjust field based on line
            if line == 'Wide Outside Off' or line == 'Outside Off':
                field_setting['catching_positions'].extend(['slip', 'gully', 'point'])
                field_setting['boundary_riders'].extend(['third man', 'deep cover'])
                field_setting['description'] = 'Off-side dominant field'
            
            elif line == 'On Stumps':
                if length == 'Short Good Length' or length == 'Short':
                    field_setting['catching_positions'].extend(['square leg', 'midwicket'])
                    field_setting['description'] = 'Short ball field with leg-side catchers'
                else:
                    field_setting['catching_positions'].extend(['mid-on', 'mid-off'])
                    field_setting['description'] = 'Straight field for wicket-to-wicket bowling'
            
            elif line == 'Down Leg' or line == 'Wide Down Leg':
                field_setting['catching_positions'].extend(['square leg', 'fine leg'])
                field_setting['boundary_riders'].extend(['deep square leg', 'deep fine leg'])
                field_setting['description'] = 'Leg-side dominant field'
                
            # Adjust based on length
            if length == 'Full Toss' or length == 'Yorker':
                field_setting['catching_positions'].extend(['mid-on', 'mid-off'])
                field_setting['description'] += ' with straight catchers for full bowling'
            elif length == 'Good Length':
                field_setting['catching_positions'].extend(['silly point', 'short leg'])
                field_setting['description'] += ' with close-in catchers'
        
        # Phase-specific adjustments
        if phase == 1:  # Powerplay
            field_setting['boundary_riders'] = ['deep square leg', 'deep point', 'long-off', 'deep fine leg']
            field_setting['description'] += ' - Defensive field with 4 boundary riders as per powerplay restrictions'
            
        elif phase == 2:  # Middle overs
            field_setting['boundary_riders'] = ['deep square leg', 'deep midwicket', 'long-on', 'long-off', 'deep cover']
            
            if bowler_type and 'spin' in bowler_type.lower():
                field_setting['catching_positions'].extend(['slip', 'short leg', 'silly point'])
                field_setting['description'] += ' - Attacking field for spin with close catchers and boundary protection'
            else:
                field_setting['catching_positions'].extend(['slip', 'gully'])
                field_setting['description'] += ' - Balanced field with catching positions and boundary protection'
                
        elif phase == 3:  # Death overs
            field_setting['boundary_riders'] = ['deep square leg', 'deep midwicket', 'long-on', 'long-off', 'deep cover', 'third man']
            field_setting['description'] += ' - Defensive field focused on boundary protection for death overs'
        
        # Remove duplicates
        field_setting['catching_positions'] = list(set(field_setting['catching_positions']))
        field_setting['boundary_riders'] = list(set(field_setting['boundary_riders']))
        
        return field_setting
    
    def generate_complete_bowling_plan(self, batter, available_bowlers=None):
        """
        Generate a comprehensive bowling plan for a specific batter
        
        Parameters:
        -----------
        batter : str
            Batter name
            
        available_bowlers : list of dict, optional
            List of available bowlers with their types
            
        Returns:
        --------
        Dictionary with comprehensive bowling plan
        """
        # Get batter information
        if batter not in self.batter_profiles:
            return {"error": "Batter not found in dataset"}
            
        batter_info = self.batter_profiles[batter]
        
        # Identify batter weaknesses
        weaknesses = self.find_batter_weaknesses(batter)
        
        # Identify optimal matchups
        matchups = self.identify_optimal_matchups(batter, available_bowlers)
        
        # Generate phase-specific plans
        if available_bowlers:
            bowler_types = [b.get('type', b.get('bowl_kind', 'Unknown')) for b in available_bowlers]
            bowler_types = [t for t in bowler_types if t != 'Unknown']
        else:
            bowler_types = list(self.bowler_profiles.keys())
        
        phase_plans = self.generate_phase_specific_plan(batter, bowler_types)
        
        # Combine all insights into a comprehensive plan
        comprehensive_plan = {
            'batter': batter,
            'batter_info': {
                'bat_hand': batter_info.get('bat_hand', 'Unknown'),
                'strike_rate': batter_info.get('strike_rate', 0),
                'average': batter_info.get('average', 0),
                'total_balls_analyzed': batter_info.get('total_balls', 0)
            },
            'weaknesses': weaknesses.get('weaknesses', []),
            'optimal_matchups': matchups.get('matchups', []),
            'phase_plans': phase_plans.get('phase_plans', {}),
            'summary': self._generate_plan_summary(batter, weaknesses, matchups, phase_plans)
        }
        
        return comprehensive_plan
    
    def _generate_plan_summary(self, batter, weaknesses, matchups, phase_plans):
        """Generate a textual summary of the bowling plan"""
        summary = []
        
        # Overall strategy
        summary.append(f"BOWLING PLAN FOR {batter.upper()}")
        
        # Batter hand
        bat_hand = self.batter_profiles.get(batter, {}).get('bat_hand', 'Unknown')
        if bat_hand in ['LHB', 'RHB']:
            hand_text = "Left-handed" if bat_hand == 'LHB' else "Right-handed"
            summary.append(f"Batter type: {hand_text}")
        
        # Key weaknesses
        if weaknesses and weaknesses.get('weaknesses'):
            summary.append("\nKEY WEAKNESSES:")
            for w in weaknesses['weaknesses']:
                if w['type'] == 'bowler_type':
                    summary.append(f"- Vulnerable against {w['bowler_type']} (Avg: {w['average']:.2f})")
                elif w['type'] == 'line_length':
                    summary.append(f"- Struggles against {w['line']} {w['length']} deliveries (Avg: {w['average']:.2f})")
                elif w['type'] == 'phase':
                    summary.append(f"- Less effective in {w['phase']} phase (SR: {w['strike_rate']:.2f})")
        
        # Best matchups
        if matchups and matchups.get('matchups') and len(matchups['matchups']) > 0:
            summary.append("\nOPTIMAL BOWLER MATCHUPS:")
            for i, m in enumerate(matchups['matchups'][:2], 1):
                summary.append(f"{i}. {m['bowler']} ({m['type']})")
                if m['recommendations'] and len(m['recommendations']) > 0:
                    rec = m['recommendations'][0]
                    summary.append(f"   Line: {rec['line']}, Length: {rec['length']}")
        
        # Phase strategies
        if phase_plans and phase_plans.get('phase_plans'):
            summary.append("\nPHASE-SPECIFIC STRATEGIES:")
            for phase_id, plan in phase_plans['phase_plans'].items():
                summary.append(f"\n{plan['phase']}:")
                summary.append(f"- Preferred bowler type: {plan['optimal_bowler_type']}")
                
                if plan['line_length_recommendations'] and len(plan['line_length_recommendations']) > 0:
                    rec = plan['line_length_recommendations'][0]
                    summary.append(f"- Bowl {rec['line']} {rec['length']}")
                
                summary.append(f"- Field setting: {plan['field_setting']['description']}")
        
        return "\n".join(summary)