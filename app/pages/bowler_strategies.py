"""Bowler Strategies Page"""
import sys
from pathlib import Path
import streamlit as st

from app.components.metrics import display_bowler_metrics
from app.components.visualizations import display_bowler_effectiveness
from app.components.selectors import (
    select_bowler,
    select_phase
)
from app import config

def app(state_container):
    """Render the bowler strategies page
    
    Parameters:
        state_container (StateContainer): Application state container
    """
    st.title("Bowler Strategies")
    
    # Initialize analyzer if needed
    if not state_container.analyzers_initialized:
        st.error("Please initialize analyzers first")
        return
    
    # Bowler selection
    selected_bowler = select_bowler(state_container)
    
    if not selected_bowler:
        st.warning("Please select a bowler to analyze")
        return
    
    # Get bowler profile
    bowler_profile = state_container.bowler_analyzer.get_bowler_profile(selected_bowler)
    
    # Display overall metrics
    st.header("Overall Statistics")
    display_bowler_metrics(bowler_profile)
    
    # Phase-wise analysis
    st.header("Phase Analysis")
    phase = select_phase()
    
    # Get line-length stats for selected phase
    line_length_stats = state_container.bowler_analyzer.get_phase_line_length_stats(
        selected_bowler,
        int(phase)
    )
    
    if line_length_stats:
        st.subheader(f"Line and Length Analysis - {config.PHASE_NAMES[int(phase)]}")
        display_bowler_effectiveness(line_length_stats)
        
        # Display successful patterns
        patterns = state_container.bowler_analyzer.get_successful_patterns(
            selected_bowler,
            int(phase)
        )
        
        if patterns:
            st.subheader("Successful Bowling Patterns")
            for pattern in patterns:
                st.write(f"• {pattern}")
    else:
        st.warning(f"Not enough data for {selected_bowler} in {config.PHASE_NAMES[int(phase)]}")
    
    # Display recommendations
    st.header("Recommendations")
    recommendations = state_container.bowler_analyzer.get_recommendations(
        selected_bowler,
        int(phase)
    )
    
    if recommendations:
        for rec in recommendations:
            st.write(f"• {rec}")
    else:
        st.info("No specific recommendations available based on current data")