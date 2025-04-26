"""Smart Stats Analysis page for T20 Cricket Analysis"""
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

def app(state):
    """Smart Stats Analysis page"""
    st.title("🏏 Smart Stats Analysis")
    st.markdown("""
    This page analyzes T20 cricket data using ESPNcricinfo's Smart Stats metrics that account for:
    - **Pressure Index**: Contextual difficulty of scoring runs
    - **Smart Runs**: Quality-adjusted batting performance
    - **Smart Wickets**: Quality-adjusted bowling performance
    - **Player Impact**: Overall contribution to match outcome
    """)
    
    # Initialize analyzer if needed
    if not hasattr(state, 'smart_stats_analyzer'):
        try:
            state.smart_stats_analyzer = SmartStatsAnalyzer()
        except Exception as e:
            st.error(f"Error initializing Smart Stats Analyzer: {e}")
            st.info("Please run the backend processor with Smart Stats enabled first.")
            return
    
    analyzer = state.smart_stats_analyzer
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Smart Runs", 
        "Smart Wickets", 
        "Player Impact",
        "Key Moments"
    ])
    
    # Smart Runs Analysis
    with tab1:
        st.header("Smart Runs Analysis")
        try:
            rankings = analyzer.get_smart_runs_ranking()
            smart_runs_df = pd.DataFrame(rankings).set_index('player')
            st.dataframe(smart_runs_df)
            if not smart_runs_df.empty:
                st.subheader("Top Batsmen by Smart Runs")
                fig, ax = plt.subplots(figsize=(10, 6))
                top_10 = smart_runs_df.nlargest(10, 'smart_runs')
                sns.barplot(data=top_10.reset_index(), x='player', y='smart_runs')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Error fetching Smart Runs data: {e}")
    
    # Smart Wickets Analysis
    with tab2:
        st.header("Smart Wickets Analysis")
        try:
            rankings = analyzer.get_smart_wickets_ranking()
            smart_wickets_df = pd.DataFrame(rankings).set_index('player')
            st.dataframe(smart_wickets_df)
            if not smart_wickets_df.empty:
                st.subheader("Top Bowlers by Smart Wickets")
                fig, ax = plt.subplots(figsize=(10, 6))
                top_10 = smart_wickets_df.nlargest(10, 'smart_wickets')
                sns.barplot(data=top_10.reset_index(), x='player', y='smart_wickets')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Error fetching Smart Wickets data: {e}")
    
    # Player Impact Analysis
    with tab3:
        st.header("Player Impact Analysis")
        try:
            rankings = analyzer.get_player_impact_ranking()
            player_impact_df = pd.DataFrame(rankings).set_index('player')
            st.dataframe(player_impact_df)
            if not player_impact_df.empty:
                st.subheader("Top Players by Impact Score")
                fig, ax = plt.subplots(figsize=(10, 6))
                top_10 = player_impact_df.nlargest(10, 'total_impact')
                sns.barplot(data=top_10.reset_index(), x='player', y='total_impact')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Error fetching Player Impact data: {e}")
    
    # Key Moments Analysis
    with tab4:
        st.header("Key Moments Analysis")
        try:
            # Show match selector
            matches = analyzer.get_all_matches()
            selected_match = st.selectbox("Select Match", matches)
            
            if selected_match:
                key_moments = analyzer.get_key_moments(selected_match)
                key_moments_df = pd.DataFrame(key_moments)
                if not key_moments_df.empty:
                    st.dataframe(key_moments_df)
                else:
                    st.info("No key moments found for this match.")
        except Exception as e:
            st.error(f"Error fetching Key Moments data: {e}")