"""Matchup Optimizer Page"""
import sys
from pathlib import Path
import streamlit as st

from app.components.metrics import display_batter_metrics, display_bowler_metrics
from app.components.visualizations import display_vulnerability_heatmap
from app.components.selectors import (
    select_batter,
    select_bowler,
    select_phase
)
from app import config

def app(state_container):
    """Render the matchup optimizer page
    
    Parameters:
        state_container (StateContainer): Application state container
    """
    st.title("Match-up Optimizer")
    
    # Initialize analyzer if needed
    if not state_container.analyzers_initialized:
        st.error("Please initialize analyzers first")
        return
    
    # Player selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Batter")
        selected_batter = select_batter(state_container)
        if selected_batter:
            batter_profile = state_container.batter_analyzer.get_batter_profile(selected_batter)
            display_batter_metrics(batter_profile)
    
    with col2:
        st.header("Bowler")
        selected_bowler = select_bowler(state_container)
        if selected_bowler:
            bowler_profile = state_container.bowler_analyzer.get_bowler_profile(selected_bowler)
            display_bowler_metrics(bowler_profile)
    
    if not (selected_batter and selected_bowler):
        st.warning("Please select both a batter and a bowler to analyze matchup")
        return
    
    # Phase selection
    st.header("Phase Analysis")
    phase = select_phase()
    
    # Get matchup analysis
    matchup = state_container.plan_generator.analyze_matchup(
        selected_batter,
        selected_bowler,
        int(phase)
    )
    
    if not matchup:
        st.warning("Not enough data for detailed matchup analysis")
        return
    
    # Display head-to-head stats
    st.subheader("Head-to-Head Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Balls", matchup['total_balls'])
        st.metric("Runs", matchup['total_runs'])
    
    with col2:
        st.metric("Strike Rate", f"{matchup['strike_rate']:.2f}")
        st.metric("Dot Ball %", f"{matchup['dot_percentage']:.1f}%")
    
    with col3:
        st.metric("Boundaries", matchup['boundaries'])
        st.metric("Dismissals", matchup['dismissals'])
    
    # Display matchup insights
    st.subheader("Matchup Insights")
    for insight in matchup.get('insights', []):
        st.write(f"• {insight}")
    
    # Display recommended plans
    st.subheader("Recommended Plans")
    plans = matchup.get('recommended_plans', [])
    
    if plans:
        for i, plan in enumerate(plans, 1):
            with st.expander(f"Plan {i}"):
                st.write(f"**Strategy:** {plan['strategy']}")
                st.write(f"**Expected Outcome:** {plan['expected_outcome']}")
                st.write("**Key Points:**")
                for point in plan['key_points']:
                    st.write(f"• {point}")
    else:
        st.info("No specific plans available for this matchup")