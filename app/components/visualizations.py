"""Wrapper components for visualizations"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import streamlit as st


from app.utils.visualization import (
    create_vulnerability_heatmap,
    create_field_placement_visualization,
    get_strategy_visualization,
    create_bowler_economy_heatmap,
    create_bowler_strike_rate_heatmap,
    create_phase_vulnerability_heatmap,
    create_style_vulnerability_heatmap
)
from app import config

def display_vulnerability_heatmap(profile_or_rankings):
    """Display the overall vulnerability heatmap
    
    Parameters:
        profile_or_rankings: Either a batter profile or vulnerability rankings dictionary
    """
    if not profile_or_rankings:
        st.info("No vulnerability data available")
        return

    # If we received vulnerability rankings, we need the line/length data
    if isinstance(profile_or_rankings, dict) and 'vs_line_length' in profile_or_rankings:
        # This is a rankings dictionary, use it directly
        fig = create_vulnerability_heatmap(profile_or_rankings)
        st.pyplot(fig)
    else:
        st.error("Unable to create vulnerability heatmap - missing line/length data")

def display_bowler_effectiveness(line_length_stats):
    """Display bowler effectiveness heatmaps"""
    if not line_length_stats:
        st.write("Not enough data to visualize line and length effectiveness.")
        return
        
    # Create separate heatmaps for economy and strike rate
    tab1, tab2 = st.tabs(["Economy Rate", "Strike Rate"])
    
    with tab1:
        economy_fig = create_bowler_economy_heatmap(line_length_stats)
        st.pyplot(economy_fig)
    
    with tab2:
        strike_rate_fig = create_bowler_strike_rate_heatmap(line_length_stats)
        st.pyplot(strike_rate_fig)

def display_field_setting(field_settings):
    """Display field setting visualization with error handling"""
    try:
        field_fig = create_field_placement_visualization(field_settings)
        st.pyplot(field_fig)
    except Exception as e:
        st.error(f"Error creating field visualization: {e}")
        st.write("Could not create field visualization.")

def display_phase_heatmap(phase_stats):
    """Display phase-specific vulnerability heatmap"""
    if not phase_stats:
        st.info("No phase-specific data available")
        return
        
    # Handle both profile and rankings data structures
    if isinstance(phase_stats, dict):
        if 'vs_line_length' in phase_stats:
            fig = create_phase_vulnerability_heatmap(phase_stats)
            st.pyplot(fig)
        else:
            st.error("Unable to create phase heatmap - missing line/length data")

def display_style_heatmap(style_stats):
    """Display bowling style specific vulnerability heatmap"""
    if not style_stats:
        st.info("No bowling style data available")
        return
        
    # Handle both profile and rankings data structures
    if isinstance(style_stats, dict):
        if 'vs_line_length' in style_stats:
            fig = create_style_vulnerability_heatmap(style_stats)
            st.pyplot(fig)
        else:
            st.error("Unable to create style heatmap - missing line/length data")

def display_plan_summary(plan):
    """Display bowling plan summary visualization"""
    try:
        summary_fig = get_strategy_visualization(plan)
        st.pyplot(summary_fig)
    except Exception as e:
        st.error(f"Error creating summary visualization: {e}")
        st.write("Could not create plan summary visualization.")