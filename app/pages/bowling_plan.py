"""Complete Bowling Plan Page"""
import sys
from pathlib import Path
import streamlit as st

from app.components.metrics import display_batter_metrics
from app.components.visualizations import display_plan_summary
from app.components.selectors import (
    select_batter,
    select_bowling_style,
    select_phase_emphasis,
    select_strategy_emphasis
)
from app import config

def app(state_container):
    """Render the complete bowling plan page
    
    Parameters:
        state_container (StateContainer): Application state container
    """
    st.title("Complete Bowling Plan")
    
    # Initialize analyzer if needed
    if not state_container.analyzers_initialized:
        st.error("Please initialize analyzers first")
        return
    
    # Batter selection
    st.header("Target Batter")
    selected_batter = select_batter(state_container)
    
    if not selected_batter:
        st.warning("Please select a batter to generate plan")
        return
    
    # Display batter profile
    batter_profile = state_container.batter_analyzer.get_batter_profile(selected_batter)
    display_batter_metrics(batter_profile)
    
    # Plan configuration
    st.header("Plan Configuration")
    
    # Filter bowler types
    selected_types = select_bowling_style(state_container)
    
    # Phase and strategy emphasis
    col1, col2 = st.columns(2)
    
    with col1:
        phase_emphasis = select_phase_emphasis()
    
    with col2:
        strategy_emphasis = select_strategy_emphasis()
    
    # Generate plan button
    if st.button("Generate Bowling Plan"):
        with st.spinner("Generating optimal bowling plan..."):
            plan = state_container.plan_generator.generate_complete_plan(
                selected_batter,
                bowler_types=selected_types or None,
                phase_emphasis=phase_emphasis,
                strategy_emphasis=strategy_emphasis
            )
            
            if not plan:
                st.error("Could not generate plan with current configuration")
                return
            
            # Display plan summary visualization
            st.header("Plan Summary")
            display_plan_summary(plan)
            
            # Display phase-wise plans
            st.header("Phase-wise Plan")
            for phase_num in range(1, 5):
                phase_plan = plan.get(f'phase_{phase_num}', {})
                if phase_plan:
                    with st.expander(f"{config.PHASE_NAMES[phase_num]}", expanded=True):
                        # Primary bowlers
                        st.subheader("Primary Bowlers")
                        for bowler in phase_plan.get('primary_bowlers', []):
                            st.write(f"• {bowler['name']} ({bowler['overs']} overs)")
                            st.write(f"  Strategy: {bowler['strategy']}")
                        
                        # Backup options
                        st.subheader("Backup Options")
                        for bowler in phase_plan.get('backup_bowlers', []):
                            st.write(f"• {bowler['name']}")
                            st.write(f"  Situation: {bowler['situation']}")
                        
                        # Key points
                        st.subheader("Key Points")
                        for point in phase_plan.get('key_points', []):
                            st.write(f"• {point}")
            
            # Display overall recommendations
            st.header("Overall Recommendations")
            for rec in plan.get('overall_recommendations', []):
                st.write(f"• {rec}")
            
            # Field placement suggestions
            if 'field_settings' in plan:
                st.header("Field Placement Suggestions")
                for phase, setting in plan['field_settings'].items():
                    with st.expander(f"Field Setting - {phase}"):
                        st.write(f"**Base Setting:** {setting['base']}")
                        st.write("**Variations:**")
                        for var in setting['variations']:
                            st.write(f"• {var}")