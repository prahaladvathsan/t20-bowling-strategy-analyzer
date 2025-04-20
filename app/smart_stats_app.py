import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent.parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

# Import modules
from smart_stats_analyzer import SmartStatsAnalyzer

def app():
    st.title("🏏 Smart Stats Analysis")
    st.markdown("""
    This page analyzes T20 cricket data using ESPNcricinfo's Smart Stats metrics that account for:
    - **Pressure Index**: Contextual difficulty of scoring runs
    - **Smart Runs**: Quality-adjusted batting performance
    - **Smart Wickets**: Quality-adjusted bowling performance
    - **Player Impact**: Overall contribution to match outcome
    """)
    
    # Initialize analyzer
    try:
        analyzer = SmartStatsAnalyzer()
    except Exception as e:
        st.error(f"Error initializing Smart Stats Analyzer: {e}")
        st.info("Please run the backend processor with Smart Stats enabled first.")
        return
        
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Smart Runs", 
        "Smart Wickets", 
        "Player Impact",
        "Key Moments"
    ])
    
    with tab1:
        st.header("Smart Runs Analysis")
        st.markdown("""
        Smart Runs factors in the match situation, quality of bowling, and pressure when runs are scored.
        This provides a more nuanced measure of batting performance compared to traditional runs.
        """)
        
        # Get all matches
        all_matches = analyzer.get_all_matches()
        match_options = ["All Matches"] + all_matches
        
        selected_match = st.selectbox(
            "Select Match", 
            options=match_options,
            format_func=lambda x: f"Match {x}" if x != "All Matches" else x
        )
        
        # Get ranking
        match_id = None if selected_match == "All Matches" else selected_match
        smart_runs_ranking = analyzer.get_smart_runs_ranking(match_id)
        
        if smart_runs_ranking:
            # Create DataFrame for display
            df = pd.DataFrame(smart_runs_ranking[:20])  # Top 20 for readability
            
            # Show table
            st.subheader("Smart Runs Leaderboard")
            st.dataframe(
                df[['player', 'smart_runs', 'conventional_runs', 'balls_faced', 'smart_sr', 'conventional_sr']],
                column_config={
                    'player': 'Batsman',
                    'smart_runs': 'Smart Runs',
                    'conventional_runs': 'Conventional Runs',
                    'balls_faced': 'Balls Faced',
                    'smart_sr': 'Smart Strike Rate',
                    'conventional_sr': 'Conventional Strike Rate'
                },
                use_container_width=True
            )
            
            # Visualization
            st.subheader("Smart Runs vs Conventional Runs")
            
            # Filter to top 10 for visualization
            top_batsmen = df.head(10)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot bars
            x = range(len(top_batsmen))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], top_batsmen['conventional_runs'], width, label='Conventional Runs')
            ax.bar([i + width/2 for i in x], top_batsmen['smart_runs'], width, label='Smart Runs')
            
            # Add labels and legend
            ax.set_xlabel('Batsman')
            ax.set_ylabel('Runs')
            ax.set_title('Conventional vs Smart Runs Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(top_batsmen['player'], rotation=45, ha='right')
            ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Individual batsman analysis
            st.subheader("Individual Batsman Analysis")
            
            selected_batsman = st.selectbox(
                "Select Batsman",
                options=df['player'].tolist()
            )
            
            if selected_batsman:
                batsman_stats = analyzer.get_batsman_smart_stats(batsman_name=selected_batsman)
                
                if batsman_stats and batsman_stats['matches'] > 0:
                    # Display overall stats
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Smart Runs", f"{batsman_stats['smart_runs_total']:.2f}")
                    with col2:
                        st.metric("Smart Strike Rate", f"{batsman_stats['smart_sr_overall']:.2f}")
                    with col3:
                        st.metric("Matches", batsman_stats['matches'])
                    
                    # Match-by-match performance
                    st.subheader("Match Performances")
                    
                    # Create DataFrame
                    performances_df = pd.DataFrame(batsman_stats['match_performances'])
                    
                    # Show table
                    st.dataframe(
                        performances_df,
                        column_config={
                            'match_id': 'Match',
                            'smart_runs': 'Smart Runs',
                            'conventional_runs': 'Conventional Runs',
                            'balls_faced': 'Balls Faced',
                            'smart_sr': 'Smart SR',
                            'conventional_sr': 'Conv. SR',
                            'impact': 'Impact'
                        },
                        use_container_width=True
                    )
                    
                    # Plot performance trend
                    if len(performances_df) > 1:
                        st.subheader("Performance Trend")
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        performances_df['match_num'] = range(1, len(performances_df) + 1)
                        
                        # Plot lines
                        ax.plot(performances_df['match_num'], performances_df['conventional_runs'], 'b-', label='Conventional Runs')
                        ax.plot(performances_df['match_num'], performances_df['smart_runs'], 'r-', label='Smart Runs')
                        
                        # Add labels
                        ax.set_xlabel('Match Number')
                        ax.set_ylabel('Runs')
                        ax.set_title(f'{selected_batsman} - Performance Trend')
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        
                        st.pyplot(fig)
                else:
                    st.write("No detailed stats available for this batsman.")
        else:
            st.info("No smart runs data available for the selected match.")
    
    with tab2:
        st.header("Smart Wickets Analysis")
        st.markdown("""
        Smart Wickets values each dismissal based on the quality of the batsman dismissed,
        the match situation, and impact on the match outcome.
        """)
        
        # Match selection
        selected_match = st.selectbox(
            "Select Match", 
            options=match_options,
            key="wickets_match_selector",
            format_func=lambda x: f"Match {x}" if x != "All Matches" else x
        )
        
        # Get ranking
        match_id = None if selected_match == "All Matches" else selected_match
        smart_wickets_ranking = analyzer.get_smart_wickets_ranking(match_id)
        
        if smart_wickets_ranking:
            # Create DataFrame for display
            df = pd.DataFrame(smart_wickets_ranking[:20])  # Top 20 for readability
            
            # Show table
            st.subheader("Smart Wickets Leaderboard")
            st.dataframe(
                df[['player', 'smart_wickets', 'conventional_wickets', 'total_overs', 'smart_er', 'conventional_er']],
                column_config={
                    'player': 'Bowler',
                    'smart_wickets': 'Smart Wickets',
                    'conventional_wickets': 'Wickets',
                    'total_overs': 'Overs',
                    'smart_er': 'Smart Economy',
                    'conventional_er': 'Economy Rate'
                },
                use_container_width=True
            )
            
            # Visualization
            st.subheader("Smart Wickets vs Conventional Wickets")
            
            # Filter to top 10 for visualization
            top_bowlers = df.head(10)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot bars
            x = range(len(top_bowlers))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], top_bowlers['conventional_wickets'], width, label='Conventional Wickets')
            ax.bar([i + width/2 for i in x], top_bowlers['smart_wickets'], width, label='Smart Wickets')
            
            # Add labels and legend
            ax.set_xlabel('Bowler')
            ax.set_ylabel('Wickets')
            ax.set_title('Conventional vs Smart Wickets Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(top_bowlers['player'], rotation=45, ha='right')
            ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Individual bowler analysis
            st.subheader("Individual Bowler Analysis")
            
            selected_bowler = st.selectbox(
                "Select Bowler",
                options=df['player'].tolist()
            )
            
            if selected_bowler:
                bowler_stats = analyzer.get_bowler_smart_stats(bowler_name=selected_bowler)
                
                if bowler_stats and bowler_stats['matches'] > 0:
                    # Display overall stats
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Smart Wickets", f"{bowler_stats['smart_wickets_total']:.2f}")
                    with col2:
                        st.metric("Smart Economy Rate", f"{bowler_stats['smart_er_overall']:.2f}")
                    with col3:
                        st.metric("Matches", bowler_stats['matches'])
                    
                    # Match-by-match performance
                    st.subheader("Match Performances")
                    
                    # Create DataFrame
                    performances_df = pd.DataFrame(bowler_stats['match_performances'])
                    
                    # Show table
                    st.dataframe(
                        performances_df,
                        column_config={
                            'match_id': 'Match',
                            'smart_wickets': 'Smart Wickets',
                            'conventional_wickets': 'Wickets',
                            'overs': 'Overs',
                            'runs': 'Runs',
                            'smart_er': 'Smart ER',
                            'conventional_er': 'Economy',
                            'impact': 'Impact'
                        },
                        use_container_width=True
                    )
                    
                    # Plot performance trend
                    if len(performances_df) > 1:
                        st.subheader("Performance Trend")
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        performances_df['match_num'] = range(1, len(performances_df) + 1)
                        
                        # Plot lines
                        ax.plot(performances_df['match_num'], performances_df['conventional_wickets'], 'b-', label='Conventional Wickets')
                        ax.plot(performances_df['match_num'], performances_df['smart_wickets'], 'r-', label='Smart Wickets')
                        
                        # Add labels
                        ax.set_xlabel('Match Number')
                        ax.set_ylabel('Wickets')
                        ax.set_title(f'{selected_bowler} - Performance Trend')
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        
                        st.pyplot(fig)
                else:
                    st.write("No detailed stats available for this bowler.")
        else:
            st.info("No smart wickets data available for the selected match.")
    
    with tab3:
        st.header("Player Impact Analysis")
        st.markdown("""
        Player Impact combines batting and bowling contributions to measure
        each player's overall influence on match outcomes.
        """)
        
        # Match selection
        selected_match = st.selectbox(
            "Select Match", 
            options=match_options,
            key="impact_match_selector",
            format_func=lambda x: f"Match {x}" if x != "All Matches" else x
        )
        
        # Get ranking
        match_id = None if selected_match == "All Matches" else selected_match
        player_impact_ranking = analyzer.get_player_impact_ranking(match_id)
        
        if player_impact_ranking:
            # Create DataFrame for display
            df = pd.DataFrame(player_impact_ranking[:20])  # Top 20 for readability
            
            # Show table
            st.subheader("Player Impact Leaderboard")
            st.dataframe(
                df[['player', 'total_impact', 'batting_impact', 'bowling_impact']],
                column_config={
                    'player': 'Player',
                    'total_impact': 'Total Impact',
                    'batting_impact': 'Batting Impact',
                    'bowling_impact': 'Bowling Impact'
                },
                use_container_width=True
            )
            
            # Visualization
            st.subheader("Impact Breakdown (Top 10 Players)")
            
            # Filter to top 10 for visualization
            top_players = df.head(10)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot stacked bars
            x = range(len(top_players))
            
            ax.bar(x, top_players['batting_impact'], label='Batting Impact', color='skyblue')
            ax.bar(x, top_players['bowling_impact'], bottom=top_players['batting_impact'], 
                   label='Bowling Impact', color='lightgreen')
            
            # Add labels and legend
            ax.set_xlabel('Player')
            ax.set_ylabel('Impact Score')
            ax.set_title('Player Impact Breakdown')
            ax.set_xticks(x)
            ax.set_xticklabels(top_players['player'], rotation=45, ha='right')
            ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Player comparison
            st.subheader("Player Impact Comparison")
            col1, col2 = st.columns(2)
            
            with col1:
                player1 = st.selectbox(
                    "Select Player 1",
                    options=df['player'].tolist(),
                    index=0 if len(df) > 0 else None
                )
            
            with col2:
                remaining_players = [p for p in df['player'].tolist() if p != player1]
                player2 = st.selectbox(
                    "Select Player 2",
                    options=remaining_players,
                    index=0 if len(remaining_players) > 0 else None
                )
            
            if player1 and player2:
                # Get player data
                player1_data = df[df['player'] == player1].iloc[0] if not df[df['player'] == player1].empty else None
                player2_data = df[df['player'] == player2].iloc[0] if not df[df['player'] == player2].empty else None
                
                if player1_data is not None and player2_data is not None:
                    # Create comparison chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Data for chart
                    categories = ['Total Impact', 'Batting Impact', 'Bowling Impact']
                    player1_values = [
                        player1_data['total_impact'], 
                        player1_data['batting_impact'], 
                        player1_data['bowling_impact']
                    ]
                    player2_values = [
                        player2_data['total_impact'], 
                        player2_data['batting_impact'], 
                        player2_data['bowling_impact']
                    ]
                    
                    # Bar positions
                    x = np.arange(len(categories))
                    width = 0.35
                    
                    # Plot bars
                    ax.bar(x - width/2, player1_values, width, label=player1, color='cornflowerblue')
                    ax.bar(x + width/2, player2_values, width, label=player2, color='lightcoral')
                    
                    # Add labels
                    ax.set_xlabel('Impact Category')
                    ax.set_ylabel('Impact Score')
                    ax.set_title('Player Comparison')
                    ax.set_xticks(x)
                    ax.set_xticklabels(categories)
                    ax.legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
        else:
            st.info("No player impact data available for the selected match.")
    
    with tab4:
        st.header("Key Match Moments")
        st.markdown("""
        Significant events that had the largest impact on match outcomes,
        based on changes in win probability.
        """)
        
        # Match selection (required for key moments)
        all_matches_except_all = analyzer.get_all_matches()
        selected_match = st.selectbox(
            "Select Match", 
            options=all_matches_except_all,
            key="moments_match_selector",
            format_func=lambda x: f"Match {x}"
        )
        
        if selected_match:
            key_moments = analyzer.get_key_moments(selected_match)
            
            if key_moments:
                # Create DataFrame for display
                df = pd.DataFrame(key_moments)
                
                # Show table
                st.subheader("Key Moments")
                st.dataframe(
                    df[['over', 'player', 'description', 'total_impact']],
                    column_config={
                        'over': 'Over',
                        'player': 'Player',
                        'description': 'Description',
                        'total_impact': 'Impact'
                    },
                    use_container_width=True
                )
                
                # Visualization
                st.subheader("Match Impact Timeline")
                
                # Create plot
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Sort by over for timeline
                df_sorted = df.sort_values('over')
                
                # Plot points
                scatter = ax.scatter(df_sorted['over'], df_sorted['total_impact'], 
                          s=100, alpha=0.7, 
                          c=df_sorted['total_impact'].apply(lambda x: 'green' if x > 0 else 'red'))
                
                # Add labels
                for i, row in df_sorted.iterrows():
                    ax.annotate(row['player'], 
                                (row['over'], row['total_impact']),
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=8)
                
                # Draw horizontal line at y=0
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                
                # Add match progression line
                ax.plot(df_sorted['over'], df_sorted['total_impact'], 'k--', alpha=0.3)
                
                # Add labels and formatting
                ax.set_xlabel('Over')
                ax.set_ylabel('Impact on Match')
                ax.set_title('Key Moments Timeline')
                ax.grid(True, alpha=0.3)
                
                # Add legend
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Positive Impact'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Negative Impact')
                ]
                ax.legend(handles=legend_elements, loc='upper right')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Moment details
                st.subheader("Moment Details")
                
                selected_moment = st.selectbox(
                    "Select a key moment",
                    options=range(len(df)),
                    format_func=lambda x: df.iloc[x]['description']
                )
                
                if selected_moment is not None:
                    moment = df.iloc[selected_moment]
                    
                    # Display moment details
                    st.write(f"**Player:** {moment['player']}")
                    st.write(f"**Over:** {moment['over']}")
                    st.write(f"**Description:** {moment['description']}")
                    st.write(f"**Impact Value:** {moment['total_impact']:.4f}")
                    
                    if 'batting_impact' in moment and 'bowling_impact' in moment:
                        # Create impact breakdown chart
                        impact_types = ['Batting Impact', 'Bowling Impact']
                        impact_values = [moment['batting_impact'], moment['bowling_impact']]
                        
                        # Only show non-zero values
                        filtered_types = [t for t, v in zip(impact_types, impact_values) if v != 0]
                        filtered_values = [v for v in impact_values if v != 0]
                        
                        if filtered_types:
                            fig, ax = plt.subplots(figsize=(8, 4))
                            colors = ['skyblue' if v > 0 else 'lightcoral' for v in filtered_values]
                            ax.bar(filtered_types, filtered_values, color=colors)
                            ax.set_title('Impact Breakdown')
                            ax.set_ylabel('Impact Value')
                            ax.grid(axis='y', alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
            else:
                st.info("No key moments data available for the selected match.")
        else:
            st.info("Please select a match to view key moments.")

# Run the app
if __name__ == "__main__":
    app()