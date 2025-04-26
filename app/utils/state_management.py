"""State management utilities for the T20 Bowling Strategy Analyzer"""
import sys
from pathlib import Path
import streamlit as st

# Page names moved from config.py
PAGE_BATTER_ANALYSIS = "Batter Analysis"
PAGE_BOWLER_STRATEGIES = "Bowler Strategies"
PAGE_MATCHUP_OPTIMIZATION = "Matchup Optimizer"
PAGE_BOWLING_PLAN = "Bowling Plan"
PAGE_SMART_STATS = "Smart Stats Analysis"

class StateContainer:
    """Container for managing application state"""
    
    def __init__(self):
        """Initialize state container"""
        if 'initialized' not in st.session_state:
            self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize all session state variables"""
        st.session_state.initialized = True
        st.session_state.batter_analyzer = None
        st.session_state.bowler_analyzer = None
        st.session_state.plan_generator = None
        st.session_state.smart_stats_analyzer = None
        st.session_state.current_page = PAGE_BATTER_ANALYSIS
        st.session_state.app_started = False
        st.session_state.analyzers_initialized = False
    
    @property
    def current_page(self):
        """Get current page"""
        return st.session_state.current_page
    
    @current_page.setter
    def current_page(self, value):
        """Set current page"""
        st.session_state.current_page = value
    
    @property
    def batter_analyzer(self):
        """Get batter analyzer"""
        return st.session_state.batter_analyzer
    
    @batter_analyzer.setter
    def batter_analyzer(self, value):
        """Set batter analyzer"""
        st.session_state.batter_analyzer = value
    
    @property
    def bowler_analyzer(self):
        """Get bowler analyzer"""
        return st.session_state.bowler_analyzer
    
    @bowler_analyzer.setter 
    def bowler_analyzer(self, value):
        """Set bowler analyzer"""
        st.session_state.bowler_analyzer = value
    
    @property
    def plan_generator(self):
        """Get plan generator"""
        return st.session_state.plan_generator
    
    @plan_generator.setter
    def plan_generator(self, value):
        """Set plan generator"""
        st.session_state.plan_generator = value
    
    @property
    def smart_stats_analyzer(self):
        """Get smart stats analyzer"""
        return st.session_state.smart_stats_analyzer
    
    @smart_stats_analyzer.setter
    def smart_stats_analyzer(self, value):
        """Set smart stats analyzer"""
        st.session_state.smart_stats_analyzer = value
    
    @property
    def analyzers_initialized(self):
        """Check if analyzers are initialized"""
        return st.session_state.analyzers_initialized
    
    @analyzers_initialized.setter
    def analyzers_initialized(self, value):
        """Set analyzers initialized state"""
        st.session_state.analyzers_initialized = value
    
    @property
    def app_started(self):
        """Check if app is started"""
        return st.session_state.app_started
    
    @app_started.setter
    def app_started(self, value):
        """Set app started state"""
        st.session_state.app_started = value