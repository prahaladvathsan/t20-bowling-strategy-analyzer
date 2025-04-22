"""Common selection widgets for the application"""
import sys
from pathlib import Path
import streamlit as st


from app import config

def select_batter(state_container):
    """Select a batter from available options
    
    Parameters:
        state_container (StateContainer): Application state container
        
    Returns:
        str: Selected batter name or None
    """
    if not state_container.batter_analyzer:
        return None
        
    batters = state_container.batter_analyzer.get_all_batters()
    return st.selectbox(
        "Select Batter",
        options=batters,
        key="batter_selector"
    )

def select_bowler(state_container):
    """Select a bowler from available options
    
    Parameters:
        state_container (StateContainer): Application state container
        
    Returns:
        str: Selected bowler name or None
    """
    if not state_container.bowler_analyzer:
        return None
        
    bowlers = state_container.bowler_analyzer.get_all_bowlers()
    return st.selectbox(
        "Select Bowler",
        options=bowlers,
        key="bowler_selector"
    )

def select_phase():
    """Select a game phase
    
    Returns:
        str: Selected phase number (1-4)
    """
    phases = {str(k): v for k, v in config.PHASE_NAMES.items()}
    return st.selectbox(
        "Select Phase",
        options=list(phases.keys()),
        format_func=lambda x: phases[x],
        key="phase_selector"
    )

def select_bowling_style(state_container, batter=None):
    """Select a bowling style
    
    Parameters:
        state_container (StateContainer): Application state container
        batter (str, optional): If provided, only show styles for this batter
        
    Returns:
        str: Selected bowling style or None
    """
    if not state_container.batter_analyzer:
        return None
        
    styles = state_container.batter_analyzer.get_available_bowl_styles()
    if not styles:
        return None
        
    return st.selectbox(
        "Select Bowling Style",
        options=styles,
        key="style_selector"
    )

def select_phase_emphasis():
    """Select phase emphasis for bowling plan
    
    Returns:
        str: Selected phase emphasis
    """
    return st.selectbox(
        "Phase Emphasis",
        options=config.PHASE_EMPHASIS_OPTIONS,
        key="phase_emphasis_selector"
    )

def select_strategy_emphasis():
    """Select strategy emphasis for bowling plan
    
    Returns:
        str: Selected strategy emphasis
    """
    return st.selectbox(
        "Strategy Emphasis",
        options=config.STRATEGY_EMPHASIS_OPTIONS,
        key="strategy_emphasis_selector"
    )