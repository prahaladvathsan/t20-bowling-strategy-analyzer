"""Main application entry point"""
import os
import sys
from pathlib import Path

# Add the project root to Python path
root_dir = str(Path(__file__).parent.parent.absolute())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import streamlit as st
from utils.state_management import StateContainer, PAGE_BATTER_ANALYSIS, PAGE_BOWLER_STRATEGIES, PAGE_MATCHUP_OPTIMIZATION, PAGE_BOWLING_PLAN
from pages.batter_analysis import app as batter_analysis
from pages.bowler_strategies import app as bowler_strategies
from pages.matchup_optimizer import app as matchup_optimizer
from pages.bowling_plan import app as bowling_plan
from src.batter_analyzer import BatterAnalyzer
from src.bowler_analyzer import BowlerAnalyzer
from src.bowling_plan_generator import BowlingPlanGenerator

def initialize_analyzers(state):
    """Initialize all analyzers if not already initialized"""
    if not state.analyzers_initialized:
        with st.spinner("Initializing analyzers..."):
            state.batter_analyzer = BatterAnalyzer()
            state.bowler_analyzer = BowlerAnalyzer()
            state.plan_generator = BowlingPlanGenerator()
            state.analyzers_initialized = True
            st.success("Analyzers initialized successfully!")

def main():
    """Main application entry point"""
    # Set page config
    st.set_page_config(
        page_title="T20 Bowling Strategy Analyzer",
        page_icon="🏏",
        layout="wide"
    )
    
    # Initialize state
    state = StateContainer()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Initialize analyzers button
    if not state.analyzers_initialized:
        if st.sidebar.button("Initialize Analyzers"):
            initialize_analyzers(state)
    
    # Navigation options
    page = st.sidebar.radio(
        "Select Page",
        [
            PAGE_BATTER_ANALYSIS,
            PAGE_BOWLER_STRATEGIES,
            PAGE_MATCHUP_OPTIMIZATION,
            PAGE_BOWLING_PLAN
        ]
    )
    
    # Save current page in state
    state.current_page = page
    
    # Route to appropriate page
    if page == PAGE_BATTER_ANALYSIS:
        batter_analysis(state)
    elif page == PAGE_BOWLER_STRATEGIES:
        bowler_strategies(state)
    elif page == PAGE_MATCHUP_OPTIMIZATION:
        matchup_optimizer(state)
    elif page == PAGE_BOWLING_PLAN:
        bowling_plan(state)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "T20 Bowling Strategy Analyzer helps teams develop data-driven "
        "bowling strategies against specific batters."
    )

if __name__ == "__main__":
    main()