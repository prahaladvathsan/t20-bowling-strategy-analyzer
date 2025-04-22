"""Batter Analysis Page"""
import streamlit as st
import pandas as pd

from app.components.metrics import display_batter_metrics, display_weakness_summary
from app.components.visualizations import (
    display_vulnerability_heatmap,
    display_phase_heatmap,
    display_style_heatmap
)
from app.components.selectors import (
    select_batter,
    select_phase,
    select_bowling_style
)
from app import config

def app(state_container):
    """Render the batter analysis page"""
    st.title("Batter Analysis")
    
    if not state_container.analyzers_initialized:
        st.error("Please initialize analyzers first")
        return
        
    selected_batter = select_batter(state_container)
    
    if not selected_batter:
        st.warning("Please select a batter to analyze")
        return
        
    # Get overall batter profile
    batter_profile = state_container.batter_analyzer.analyze_batter_stats(
        selected_batter, 
        'overall'
    )
    
    if not batter_profile:
        st.error(f"No data found for batter {selected_batter}")
        return
        
    # Display overall metrics
    st.header("Overall Statistics")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        display_batter_metrics(batter_profile)
    
    with col2:
        # Get vulnerability rankings across all dimensions
        display_weakness_summary({
            'vs_bowler_styles': state_container.batter_analyzer.analyze_batter_stats(
                selected_batter, 'bowling_style'
            ),
            'by_phase': state_container.batter_analyzer.analyze_batter_stats(
                selected_batter, 'phase'
            ),
            'vs_line_length': state_container.batter_analyzer.analyze_batter_stats(
                selected_batter, 'line_length'
            )
        })
    
    # Vulnerability analysis
    st.header("Vulnerability Analysis")
    tab1, tab2, tab3 = st.tabs([
        "Overall Vulnerability",
        "Phase Analysis",
        "Bowling Style Analysis"
    ])
    
    with tab1:
        # Use line-length analysis for overall vulnerability
        line_length_stats = state_container.batter_analyzer.analyze_batter_stats(
            selected_batter, 'line_length'
        )
        display_vulnerability_heatmap({'vs_line_length': line_length_stats})
    
    with tab2:
        # First show the overall phase statistics
        pan_phase_stats = state_container.batter_analyzer.analyze_batter_stats(
            selected_batter, 
            'phase'
        )
        
        if pan_phase_stats:
            st.subheader("Phase-wise Statistics")
            # Create DataFrame for all phases
            phase_data = []
            for phase_num in sorted(pan_phase_stats.keys()):
                phase_stats = pan_phase_stats[phase_num]
                phase_data.append({
                    'Phase': f"Phase {phase_num}",
                    'Runs': phase_stats.get('runs', 0),
                    'Balls': phase_stats.get('balls', 0),
                    'Strike Rate': f"{phase_stats.get('strike_rate', 0):.2f}",
                    'Dismissals': phase_stats.get('dismissals', 0),
                    'Dot Ball %': f"{(phase_stats.get('dot_balls', 0) / phase_stats.get('balls', 1) * 100):.1f}%"
                })
            
            if phase_data:
                phase_df = pd.DataFrame(phase_data)
                st.table(phase_df)

        # Then show detailed analysis for selected phase
        selected_phase = select_phase()
        if selected_phase:
            phase_stats = state_container.batter_analyzer.analyze_batter_stats(
                selected_batter, 
                'phase_line_length',
                filters={'phase': str(selected_phase)}
            )
            if phase_stats:
                st.subheader(f"Phase {selected_phase} Detailed Analysis")
                display_phase_heatmap({'vs_line_length': phase_stats})
            else:
                st.info(f"No detailed data available for phase {selected_phase}")
    
    with tab3:
        # First show overall bowling style statistics
        bowl_style_stats = state_container.batter_analyzer.analyze_batter_stats(
            selected_batter, 
            'bowling_style'
        )
        
        if bowl_style_stats:
            st.subheader("Bowling Style-wise Statistics")
            # Create DataFrame for all bowling styles
            style_data = []
            for style, stats in bowl_style_stats.items():
                style_data.append({
                    'Bowling Style': style,
                    'Runs': stats.get('runs', 0),
                    'Balls': stats.get('balls', 0),
                    'Strike Rate': f"{stats.get('strike_rate', 0):.2f}",
                    'Average': f"{stats.get('average', float('inf')):.1f}" if stats.get('average', float('inf')) != float('inf') else 'N/A',
                    'Dismissals': stats.get('dismissals', 0),
                    'Dot Ball %': f"{(stats.get('dot_balls', 0) / stats.get('balls', 1) * 100):.1f}%"
                })
            
            if style_data:
                style_df = pd.DataFrame(style_data)
                st.table(style_df)

        # Then show detailed analysis for selected bowling style
        selected_style = select_bowling_style(state_container)
        if selected_style:
            detailed_stats = state_container.batter_analyzer.analyze_batter_stats(
                selected_batter,
                'style_line_length',
                filters={'bowl_style': selected_style}
            )
            if detailed_stats:
                st.subheader(f"Against {selected_style} - Detailed Analysis")
                display_style_heatmap({'vs_line_length': detailed_stats})
            else:
                st.info(f"No detailed data available for bowling style {selected_style}")
